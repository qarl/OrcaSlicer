#ifndef slic3r_PrintMan_CoreAdapter_hpp_
#define slic3r_PrintMan_CoreAdapter_hpp_

// Adapter between the fork's host types and the vendored standalone amplification core
// (PrintMan/Core). The host converts its already-built control cage and displacement shader
// to the core's neutral types, the core dices + displaces per Z-band (amplify_subcage_adaptive), and
// each band mesh comes back for the host to slice with its own slice_mesh_ex. The core stays
// host-agnostic; all libslic3r/pxr contact lives here.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "libslic3r/Point.hpp"                             // Transform3d, Vec3d
#include "libslic3r/TriangleMesh.hpp"                      // indexed_triangle_set, Vec3f/Vec3i
#include "libslic3r/Format/USDSubdiv.hpp"                  // usd_subdiv::Cage
#include "libslic3r/PrintMan/Engine.hpp"                   // DisplacementField, V3
#include "PrintMan/Core/band_usd.hpp"                      // printman::subdiv::Cage, Mesh, Shader, amplify_subcage_adaptive

namespace Slic3r { namespace PrintMan {

// The fork's usd_subdiv::Cage and the vendored printman::subdiv::Cage are structurally identical --
// same fields, Pt = std::array<double,3>, matching BOUNDARY_* and SHARPNESS_INFINITE -- so this is a
// plain field copy, no re-derivation. amplify_subcage_adaptive then dices the result.
inline printman::subdiv::Cage to_core_cage(const usd_subdiv::Cage &c)
{
    printman::subdiv::Cage o;
    o.verts           = c.verts;          // std::vector<std::array<double,3>>
    o.fvi             = c.fvi;
    o.foff            = c.foff;
    o.edge_crease     = c.edge_crease;    // std::map<long long,double>, same sharpness scale
    o.corner_sharp    = c.corner_sharp;
    o.fvar            = c.fvar;
    o.boundary        = c.boundary;       // BOUNDARY_* values match
    o.triangle_smooth = c.triangle_smooth;
    return o;
}

// Bridge the fork's displacement shader to the core's Shader interface. displace and max_reach come
// from the DisplacementField; the core forwards its per-face uv to the shader (the footprint dPdx/dPdy
// is passed zero -- point-sampled -- until the core carries it). Colour is NOT bridged: the host runs
// its own colour pass on each band mesh, so shade emits only a neutral Cout the host ignores.
struct BridgeShader : printman::Shader {
    const DisplacementField *disp;
    double                   reach;
    BridgeShader(const DisplacementField *d, double r) : disp(d), reach(r) {}
    double displace(const printman::V3 &p, const printman::V3 &n, const printman::V2 &uv) const override
    {
        double d = 0.0;
        if      (disp->eval_d) d = disp->eval_d(p, n, V3{{0, 0, 0}}, V3{{0, 0, 0}}, uv[0], uv[1]);
        else if (disp->eval)   d = disp->eval(p, n);
        return std::isfinite(d) ? d : 0.0;   // never propagate a NaN/Inf displacement into the geometry
    }
    std::vector<std::string> aov_names() const override { return {"Cout"}; }
    void shade(const printman::V3 &, const printman::V3 &, const printman::V2 &, printman::V3 *out) const override
    {
        out[0] = printman::V3{{0.5, 0.5, 0.5}};
    }
    double max_reach() const override { return reach; }
};

// World-space core cage from a fork cage + placement transform. Bakes m into the vertices (so the
// core dices in world space, adaptive to the real edge lengths, and displaces in world space), and
// reverses each face's winding + UV when the net orientation is flipped (the cage's authored winding
// XOR a mirroring det<0 placement), so the winding-derived normals still point outward. Mirrors the
// fork's own triangulate_cage(flip_winding) + its_flip_triangles(det<0) at the cage level.
inline printman::subdiv::Cage to_world_core_cage(const usd_subdiv::Cage &c, const Transform3d &m, bool net_reverse)
{
    printman::subdiv::Cage o = to_core_cage(c);
    for (auto &v : o.verts) { const Vec3d w = m * Vec3d(v[0], v[1], v[2]); v = {w.x(), w.y(), w.z()}; }
    if (net_reverse) {
        const bool has_uv = o.fvar.size() == o.fvi.size();
        for (int f = 0; f + 1 < int(o.foff.size()); ++f) {
            std::reverse(o.fvi.begin() + o.foff[f], o.fvi.begin() + o.foff[f + 1]);
            if (has_uv) std::reverse(o.fvar.begin() + o.foff[f], o.fvar.begin() + o.foff[f + 1]);
        }
    }
    return o;
}

// Per-triangle authored UV for the host colour pass, from the core mesh's per-vertex UV (the mean of
// the triangle's three corners). Parallel to append_core_mesh_to_its' index order. Empty if no UV.
inline std::vector<std::array<float, 2>> tri_uv_from_core_mesh(const printman::Mesh &m)
{
    std::vector<std::array<float, 2>> tri_uv;
    if (m.uv.size() != m.pos.size() || m.tri.empty())
        return tri_uv;
    tri_uv.reserve(m.tri.size());
    for (const auto &t : m.tri) {
        const auto &a = m.uv[t[0]]; const auto &b = m.uv[t[1]]; const auto &c = m.uv[t[2]];
        tri_uv.push_back({(a[0] + b[0] + c[0]) / 3.f, (a[1] + b[1] + c[1]) / 3.f});
    }
    return tri_uv;
}

// Append a core band mesh to an indexed_triangle_set (into its own vertex/index space) so the host can
// slice it with slice_mesh_ex. Float positions carry through unchanged.
inline void append_core_mesh_to_its(const printman::Mesh &m, indexed_triangle_set &its)
{
    const int base = int(its.vertices.size());
    its.vertices.reserve(its.vertices.size() + m.pos.size());
    for (const auto &p : m.pos) its.vertices.emplace_back(p[0], p[1], p[2]);
    its.indices.reserve(its.indices.size() + m.tri.size());
    for (const auto &t : m.tri) its.indices.emplace_back(base + int(t[0]), base + int(t[1]), base + int(t[2]));
}

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_CoreAdapter_hpp_
