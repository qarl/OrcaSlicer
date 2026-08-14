#ifndef slic3r_PrintMan_OslShader_hpp_
#define slic3r_PrintMan_OslShader_hpp_

// OslDisplaceShader -- a user-authored OSL shader behind the engine's DisplaceShader seam
// (Displace.hpp): (world point, unit normal) -> displacement mm. A compiled .oso is loaded, its
// output symbol resolved once, and each (P, N) is evaluated through OSL's ShadingSystem (LLVM-JIT).
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
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include <OSL/shaderglobals.h>
#include <OpenImageIO/errorhandler.h>
#include <OpenImageIO/ustring.h>

#include "libslic3r/PrintMan/Displace.hpp"   // V3, DisplaceShader

namespace Slic3r { namespace PrintMan {

class OslDisplaceShader {
    // A shader that only reads P/N needs no renderer services, so every base default suffices.
    // Stateless, so it is safe to share one instance across threads.
    struct Renderer final : public OSL::RendererServices {
        Renderer() : OSL::RendererServices(nullptr) {}
        int supports(OSL::string_view) const override { return 0; }
    };

    struct Ctx { OSL::PerThreadInfo *info = nullptr; OSL::ShadingContext *ctx = nullptr; };

public:
    // Load `shadername`.oso from `searchpath`, reading its `output` param (a float) each call.
    OslDisplaceShader(const std::string &searchpath,
                      const std::string &shadername,
                      const std::string &output = "Disp")
        : m_output(output), m_layer("layer1")
    {
        m_ss = new OSL::ShadingSystem(&m_rend, nullptr, &m_err);
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

        // Mark the output as a renderer output or the optimizer folds it away, unreadable.
        OSL::ustring outs[] = {m_output};
        m_ss->attribute(m_group.get(), "renderer_outputs",
                        OIIO::TypeDesc(OIIO::TypeDesc::STRING, 1), &outs[0]);

        // Resolve the output symbol ONCE. find_symbol needs an optimized group, so execute with
        // run=false to set it up without running the shader (the testshade idiom). The ShaderSymbol
        // is group-scoped -- reused across every thread's context via symbol_address.
        OSL::PerThreadInfo  *ti = m_ss->create_thread_info();
        OSL::ShadingContext *c  = m_ss->get_context(ti);
        OSL::ShaderGlobals   sg;
        std::memset((void *)&sg, 0, sizeof(sg));
        sg.renderer = &m_rend;
        m_ss->execute(*c, *m_group, sg, false);
        m_sym = m_ss->find_symbol(*m_group, m_layer, m_output);
        const bool is_float = m_sym && m_ss->symbol_typedesc(m_sym).basetype == OIIO::TypeDesc::FLOAT;
        m_ss->release_context(c);
        m_ss->destroy_thread_info(ti);
        if (!is_float) {
            m_group.reset();
            delete m_ss;
            m_ss = nullptr;
            throw std::runtime_error("OSL shader '" + shadername + "' has no float output '" +
                                     output + "'");
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
    // Thread-safe: each thread has its own ShadingContext; sg is a local.
    double operator()(const V3 &point, const V3 &normal,
                      const V3 &dPdx, const V3 &dPdy) const
    {
        OSL::ShadingContext *ctx = context_for_this_thread();

        OSL::ShaderGlobals sg;
        std::memset((void *)&sg, 0, sizeof(sg));   // ShaderGlobals is POD-ish; testshade does the same
        sg.P        = OSL::Vec3(float(point[0]), float(point[1]), float(point[2]));
        sg.dPdx     = OSL::Vec3(float(dPdx[0]),  float(dPdx[1]),  float(dPdx[2]));
        sg.dPdy     = OSL::Vec3(float(dPdy[0]),  float(dPdy[1]),  float(dPdy[2]));
        sg.N        = OSL::Vec3(float(normal[0]), float(normal[1]), float(normal[2]));
        sg.Ng       = sg.N;
        sg.renderer = const_cast<Renderer *>(&m_rend);
        m_ss->execute(*ctx, *m_group, sg);

        const void *adr = m_ss->symbol_address(*ctx, m_sym);
        return adr ? double(*reinterpret_cast<const float *>(adr)) : 0.0;
    }

private:
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

    Renderer                 m_rend;
    OIIO::ErrorHandler       m_err;
    OSL::ShadingSystem      *m_ss  = nullptr;
    OSL::ShaderGroupRef      m_group;
    const OSL::ShaderSymbol *m_sym = nullptr;   // resolved once in the ctor, const thereafter
    OSL::ustring             m_output;
    OSL::ustring             m_layer;

    mutable std::mutex                               m_mtx;
    mutable std::unordered_map<std::thread::id, Ctx> m_ctxs;
};

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_OslShader_hpp_
