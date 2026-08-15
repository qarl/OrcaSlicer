#ifndef slic3r_PrintMan_OslShader_hpp_
#define slic3r_PrintMan_OslShader_hpp_

// OslDisplaceShader -- a user-authored OSL shader behind the engine's DisplaceShader seam
// (Displace.hpp): (world point, unit normal) -> displacement mm, and optionally a surface colour
// (an `output color Cout`, linear RGB). A compiled .oso is loaded, its output symbols resolved once,
// and each (P, N) is evaluated through OSL's ShadingSystem (LLVM-JIT). Displacement and colour are
// independent outputs of the shader; each accessor runs the shader for the output it reads (there is
// no shared per-point eval -- call once per output you need).
//
// This header is compiled ONLY where OSL is available (the SLIC3R_OSL end-to-end test); it is NOT
// in libslic3r's source list, so the default build never sees the OSL headers. It slots behind the
// existing DisplacementField::eval, so the engine and slice_scene need no change. Validated against
// an independent C++ reference and hammered from 32 threads in printman spikes/osl.
//
// THREAD-SAFETY: the engine slices bands with tbb::parallel_for, so operator() runs concurrently.
// The group and output symbol are resolved once at construction (shared state const thereafter);
// each calling thread gets its own ShadingContext, created under a brief lock never held across
// execute(). A future full port can swap the mutex+map for tbb::enumerable_thread_specific.

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include <OSL/shaderglobals.h>
#include <OpenImageIO/errorhandler.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/ustring.h>

#include "libslic3r/PrintMan/Displace.hpp"   // V3, DisplaceShader

namespace Slic3r { namespace PrintMan {

class OslDisplaceShader {
    // Given OIIO's TextureSystem so texture() (image lookups, e.g. an authored map) resolves; every other
    // service stays defaulted (a shader that only reads P/N/uv still works). The texture system is thread-safe,
    // so one Renderer is safely shared across the slicer's threads.
    struct Renderer final : public OSL::RendererServices {
        explicit Renderer(OIIO::TextureSystem *ts) : OSL::RendererServices(ts) {}
        int supports(OSL::string_view) const override { return 0; }
    };

    struct Ctx { OSL::PerThreadInfo *info = nullptr; OSL::ShadingContext *ctx = nullptr; };

public:
    // Load `shadername`.oso from `searchpath`, resolving its displacement output `disp_output` (a
    // float) and, when the shader declares it, its colour output `color_output` (an OSL `color` = 3
    // floats). At least one of the two must resolve; a shader may carry either or both.
    OslDisplaceShader(const std::string &searchpath,
                      const std::string &shadername,
                      const std::string &disp_output  = "Disp",
                      const std::string &color_output = "Cout")
        : m_ts(OIIO::TextureSystem::create(/*shared=*/false)), m_rend(m_ts.get()),
          m_output(disp_output), m_color_output(color_output), m_layer("layer1")
    {
        // A PRIVATE texture system (shared=false), not the process-global one: its searchpath is set per
        // instance below, so two shaders loaded from different dirs cannot clobber each other's texture path.
        // OIIO 3.x create() returns a shared_ptr whose deleter tears the system down -- held in m_ts.
        // texture() image lookups resolve against the same dir as the shaders (where the .tx maps live).
        m_ts->attribute("searchpath", searchpath);
        m_ss = new OSL::ShadingSystem(&m_rend, m_ts.get(), &m_err);
        OSL::ustring sp(searchpath);
        m_ss->attribute("searchpath:shader", OIIO::TypeDesc::STRING, &sp);

        m_group = m_ss->ShaderGroupBegin("printman");
        if (!m_ss->Shader(*m_group, "surface", shadername, m_layer)) {
            m_group.reset();          // same order as the dtor: group before system
            delete m_ss;
            m_ss = nullptr;
            throw std::runtime_error("OSL Shader() failed for '" + shadername +
                                     "' (is " + shadername + ".oso on the searchpath?)");
        }
        m_ss->ShaderGroupEnd(*m_group);

        // Mark BOTH outputs as renderer outputs or the optimizer folds them away, unreadable. A name
        // the shader does not declare simply resolves to a null symbol below (absent, not an error).
        OSL::ustring outs[] = {m_output, m_color_output};
        m_ss->attribute(m_group.get(), "renderer_outputs",
                        OIIO::TypeDesc(OIIO::TypeDesc::STRING, 2), &outs[0]);

        // Resolve the output symbols ONCE. find_symbol needs an optimized group, so execute with
        // run=false to set it up without running the shader (the testshade idiom). The ShaderSymbols
        // are group-scoped -- reused across every thread's context via symbol_address.
        OSL::PerThreadInfo  *ti = m_ss->create_thread_info();
        OSL::ShadingContext *c  = m_ss->get_context(ti);
        OSL::ShaderGlobals   sg;
        std::memset((void *)&sg, 0, sizeof(sg));
        sg.renderer = &m_rend;
        m_ss->execute(*c, *m_group, sg, false);
        m_sym = m_ss->find_symbol(*m_group, m_layer, m_output);
        if (m_sym && m_ss->symbol_typedesc(m_sym).basetype != OIIO::TypeDesc::FLOAT)
            m_sym = nullptr;                                  // a non-float `Disp` is not a displacement
        m_color_sym = m_ss->find_symbol(*m_group, m_layer, m_color_output);
        if (m_color_sym) {
            const OIIO::TypeDesc td = m_ss->symbol_typedesc(m_color_sym);
            if (!(td.basetype == OIIO::TypeDesc::FLOAT && td.basevalues() == 3))
                m_color_sym = nullptr;                        // any 3-float aggregate (color/vector/point)
        }
        m_ss->release_context(c);
        m_ss->destroy_thread_info(ti);
        if (!m_sym && !m_color_sym) {
            m_group.reset();
            delete m_ss;
            m_ss = nullptr;
            throw std::runtime_error("OSL shader '" + shadername + "' has neither a float '" +
                                     disp_output + "' nor a colour '" + color_output + "' output");
        }
    }

    OslDisplaceShader(const OslDisplaceShader &)            = delete;
    OslDisplaceShader &operator=(const OslDisplaceShader &) = delete;

    ~OslDisplaceShader()
    {
        if (m_ss) {
            for (auto &kv : m_ctxs) {
                m_ss->release_context(kv.second.ctx);
                m_ss->destroy_thread_info(kv.second.info);
            }
            // The group's last ref must be released BEFORE the system: ~ShaderInstance dereferences
            // the ShadingSystem, and member dtors run after this body, so drop it here or that
            // release is a use-after-free (OSL's testshade documents the order).
            m_group.reset();
            delete m_ss;
        }
    }

    // The DisplaceShader: (world point, unit normal) -> displacement mm. Thread-safe. No sample
    // footprint is supplied, so derivative-driven ops (filterwidth/texture) see a zero footprint.
    double operator()(const V3 &point, const V3 &normal) const
    {
        return (*this)(point, normal, V3{{0, 0, 0}}, V3{{0, 0, 0}});
    }

    // As above, plus the sample footprint as the world-space derivatives of P (dPdx/dPdy -- e.g. the
    // refined face's two edge vectors), so OSL's filterwidth()/texture() can band-limit to it.
    // Thread-safe: each thread has its own ShadingContext; sg is a local. 0 if the shader has no
    // displacement output (a colour-only shader).
    double operator()(const V3 &point, const V3 &normal,
                      const V3 &dPdx, const V3 &dPdy) const
    {
        if (!m_sym) return 0.0;
        OSL::ShadingContext *ctx = run(point, normal, dPdx, dPdy);
        const void *adr = m_ss->symbol_address(*ctx, m_sym);
        return adr ? double(*reinterpret_cast<const float *>(adr)) : 0.0;
    }

    // True when the shader declares a colour output (an OSL `output color Cout`).
    bool has_color() const { return m_color_sym != nullptr; }

    // The shader's surface colour at (world point, unit normal): linear RGB, a PrintMan convention
    // (OSL fixes no colour space). {0,0,0} if the shader has no colour output. Same thread-safety and
    // footprint contract as operator(); like it, this runs the shader once -- colour and displacement
    // do not share an eval, so read each output with its own call.
    V3 color(const V3 &point, const V3 &normal) const
    {
        return color(point, normal, V3{{0, 0, 0}}, V3{{0, 0, 0}}, 0.0, 0.0);
    }
    V3 color(const V3 &point, const V3 &normal, const V3 &dPdx, const V3 &dPdy) const
    {
        return color(point, normal, dPdx, dPdy, 0.0, 0.0);
    }
    // As above, plus the authored surface texture coordinate as OSL's u/v globals, so a colour shader can
    // texture(name, u, v) an image map. {0,0,0} if the shader has no colour output.
    V3 color(const V3 &point, const V3 &normal, const V3 &dPdx, const V3 &dPdy, double u, double v) const
    {
        if (!m_color_sym) return V3{{0, 0, 0}};
        OSL::ShadingContext *ctx = run(point, normal, dPdx, dPdy, u, v);
        const void *adr = m_ss->symbol_address(*ctx, m_color_sym);
        if (!adr) return V3{{0, 0, 0}};
        const float *rgb = reinterpret_cast<const float *>(adr);
        return V3{{double(rgb[0]), double(rgb[1]), double(rgb[2])}};
    }

private:
    // Run the shader once for (point, normal, footprint) on this thread's ShadingContext; the caller
    // then reads whichever resolved output symbol it wants off that context via symbol_address.
    OSL::ShadingContext *run(const V3 &point, const V3 &normal, const V3 &dPdx, const V3 &dPdy,
                             double u = 0.0, double v = 0.0) const
    {
        OSL::ShadingContext *ctx = context_for_this_thread();
        OSL::ShaderGlobals sg;
        std::memset((void *)&sg, 0, sizeof(sg));   // ShaderGlobals is POD-ish; testshade does the same
        sg.P        = OSL::Vec3(float(point[0]), float(point[1]), float(point[2]));
        sg.dPdx     = OSL::Vec3(float(dPdx[0]),  float(dPdx[1]),  float(dPdx[2]));
        sg.dPdy     = OSL::Vec3(float(dPdy[0]),  float(dPdy[1]),  float(dPdy[2]));
        sg.N        = OSL::Vec3(float(normal[0]), float(normal[1]), float(normal[2]));
        sg.Ng       = sg.N;
        sg.u        = float(u);   // authored surface texture coordinate (OSL global u)
        sg.v        = float(v);   // authored surface texture coordinate (OSL global v)
        sg.renderer = const_cast<Renderer *>(&m_rend);
        m_ss->execute(*ctx, *m_group, sg);
        return ctx;
    }

    OSL::ShadingContext *context_for_this_thread() const
    {
        const std::thread::id id = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(m_mtx);
        auto it = m_ctxs.find(id);
        if (it == m_ctxs.end()) {
            Ctx c;
            c.info = m_ss->create_thread_info();
            c.ctx  = m_ss->get_context(c.info);
            it = m_ctxs.emplace(id, c).first;
        }
        return it->second.ctx;
    }

    std::shared_ptr<OIIO::TextureSystem> m_ts;   // owns the texture system; its raw ptr is handed to OSL
    Renderer                 m_rend;
    OIIO::ErrorHandler       m_err;
    OSL::ShadingSystem      *m_ss  = nullptr;
    OSL::ShaderGroupRef      m_group;
    const OSL::ShaderSymbol *m_sym       = nullptr;   // displacement output; resolved once, const thereafter
    const OSL::ShaderSymbol *m_color_sym = nullptr;   // colour output (null if the shader has none)
    OSL::ustring             m_output;
    OSL::ustring             m_color_output;
    OSL::ustring             m_layer;

    mutable std::mutex                               m_mtx;
    mutable std::unordered_map<std::thread::id, Ctx> m_ctxs;
};

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_OslShader_hpp_
