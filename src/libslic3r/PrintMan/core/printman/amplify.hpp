#pragma once
//
// Amplify a loaded USD cage: subdivide (carrying face-varying UV), then displace and shade per
// corner via the Shader. Per-corner vertices share positions at face edges (watertight slice); UV
// splits at authored seams. A mesh may bind several materials through UsdGeomSubsets, so the per-
// face material tag rides through subdivision and each face is emitted with its own shader; the
// per-material sub-meshes are then merged. (Memory-bounded per-band subdivision is a later
// refinement; this materializes the refined cage.)
//
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "printman/geom.hpp"
#include "printman/shader.hpp"
#include "printman/subdiv.hpp"
#include "printman/usd.hpp"

namespace printman {

inline bool is_subdividing(const std::string& s) {
    return s == "catmullClark" || s == "loop" || s == "bilinear";
}

inline subdiv::Cage subdiv_step(const subdiv::Cage& c, const std::string& s) {
    if (s == "catmullClark") return subdiv::catmull_clark(c);
    if (s == "loop") return subdiv::loop(c);
    if (s == "bilinear") return subdiv::bilinear(c);
    return c;
}

// Per-face material tags for the subdivided cage, matching subdiv.hpp's face output order so a
// tag follows its parent face exactly. Catmull-Clark and bilinear emit one child per corner (child
// k <- the face of corner k); loop emits the same C corner-triangles, then nf centre-triangles
// (child C+f <- face f). A non-subdividing scheme leaves the tags unchanged.
inline std::vector<int> child_face_tags(const subdiv::Cage& c, const std::string& s,
                                        const std::vector<int>& tag) {
    const int nf = c.nfaces(), C = int(c.fvi.size());
    if (s == "catmullClark" || s == "bilinear") {
        std::vector<int> out(C);
        for (int f = 0; f < nf; ++f) for (int k = c.foff[f]; k < c.foff[f + 1]; ++k) out[k] = tag[f];
        return out;
    }
    if (s == "loop") {
        std::vector<int> out(C + nf);
        for (int f = 0; f < nf; ++f) for (int k = c.foff[f]; k < c.foff[f + 1]; ++k) out[k] = tag[f];
        for (int f = 0; f < nf; ++f) out[C + f] = tag[f];
        return out;
    }
    return tag;
}

// Concatenate amplified parts into one mesh. Parts may carry different shaders' AOVs; align by name
// on the first non-empty part's schema (missing channels -> 0) so at least Cout survives across
// them. Triangle indices are rebased per part.
inline Mesh merge_meshes(const std::vector<Mesh>& parts) {
    Mesh out;
    for (const Mesh& p : parts) if (!p.aov_names.empty()) { out.aov_names = p.aov_names; break; }
    out.aov.resize(out.aov_names.size());
    for (const Mesh& p : parts) {
        std::uint32_t base = (std::uint32_t)out.pos.size();
        out.pos.insert(out.pos.end(), p.pos.begin(), p.pos.end());
        out.nrm.insert(out.nrm.end(), p.nrm.begin(), p.nrm.end());
        out.uv.insert(out.uv.end(), p.uv.begin(), p.uv.end());
        for (std::size_t k = 0; k < out.aov_names.size(); ++k) {
            int src = p.aov_index(out.aov_names[k]);
            if (src >= 0) out.aov[k].insert(out.aov[k].end(), p.aov[src].begin(), p.aov[src].end());
            else out.aov[k].resize(out.aov[k].size() + p.pos.size(), {0, 0, 0});
        }
        for (const auto& t : p.tri) out.tri.push_back({t[0] + base, t[1] + base, t[2] + base});
    }
    return out;
}

// Sit the model on the build plate (z=0). Done once, on the assembled model, so parts keep their
// relative heights (dropping each part independently would flatten them all onto the plate).
inline void drop_to_plate(Mesh& m) {
    float zmin = 1e30f;
    for (const auto& p : m.pos) zmin = std::min(zmin, p[2]);
    if (zmin < 1e30f) for (auto& p : m.pos) p[2] -= zmin;
}

// shaders is one Shader per uc.materials entry (parallel). uc.face_material selects one per face
// (empty -> every face uses material 0). Color-only subsets are crack-free (positions identical
// across a boundary); a boundary between differently-displacing subsets can gap -- rare, noted.
inline Mesh amplify_usd(const UsdCage& uc, const std::vector<const Shader*>& shaders, int level) {
    subdiv::Cage c = subdiv::build_cage(uc.points, uc.counts, uc.indices,
                                        uc.crease_indices, uc.crease_lengths, uc.crease_sharpnesses,
                                        uc.corner_indices, uc.corner_sharpnesses,
                                        uc.boundary, uc.triangle_smooth, uc.st);

    // Subdivide level-by-level, carrying the per-face material tag through each step.
    std::vector<int> tag = uc.face_material.empty() ? std::vector<int>(c.nfaces(), 0) : uc.face_material;
    subdiv::Cage r = std::move(c);
    if (is_subdividing(uc.subdiv_scheme))
        for (int i = 0; i < level; ++i) {
            tag = child_face_tags(r, uc.subdiv_scheme, tag);
            r = subdiv_step(r, uc.subdiv_scheme);
        }

    // per-vertex normals (Newell per face, accumulated)
    std::vector<V3> vn(r.verts.size(), V3{{0, 0, 0}});
    for (int f = 0; f < r.nfaces(); ++f) {
        int a = r.foff[f], b = r.foff[f + 1];
        V3 n{{0, 0, 0}};
        for (int i = a; i < b; ++i) {
            const auto& p0 = r.verts[r.fvi[i]]; const auto& p1 = r.verts[r.fvi[(i + 1 < b) ? i + 1 : a]];
            n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2]);
            n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0]);
            n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1]);
        }
        for (int i = a; i < b; ++i) { vn[r.fvi[i]][0] += n[0]; vn[r.fvi[i]][1] += n[1]; vn[r.fvi[i]][2] += n[2]; }
    }
    for (auto& n : vn) { double l = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if (l > 1e-12) { n[0]/=l; n[1]/=l; n[2]/=l; } }

    const int M = int(shaders.size());
    std::vector<Mesh> parts(M);
    std::vector<int> na(M);
    int maxna = 1;
    for (int m = 0; m < M; ++m) { parts[m].aov_names = shaders[m]->aov_names(); na[m] = int(parts[m].aov_names.size()); parts[m].aov.resize(na[m]); maxna = std::max(maxna, na[m]); }
    std::vector<V3> av(maxna);
    const bool hasuv = r.has_uv();
    auto emit = [&](Mesh& mm, const Shader& sh, int nak, int corner) -> std::uint32_t {
        int vid = r.fvi[corner];
        const auto& P = r.verts[vid]; const V3& N = vn[vid];
        V3 Pv{{P[0], P[1], P[2]}};
        V2 uv = hasuv ? V2{{r.fvar[corner][0], r.fvar[corner][1]}} : V2{{0, 0}};
        double d = sh.displace(Pv, N, uv);
        sh.shade(Pv, N, uv, av.data());
        std::uint32_t id = (std::uint32_t)mm.pos.size();
        mm.pos.push_back({(float)(P[0] + d * N[0]), (float)(P[1] + d * N[1]), (float)(P[2] + d * N[2])});
        mm.nrm.push_back({(float)N[0], (float)N[1], (float)N[2]});
        mm.uv.push_back({(float)uv[0], (float)uv[1]});
        for (int k = 0; k < nak; ++k) mm.aov[k].push_back({(float)av[k][0], (float)av[k][1], (float)av[k][2]});
        return id;
    };
    for (int f = 0; f < r.nfaces(); ++f) {
        int mi = (f < int(tag.size())) ? tag[f] : 0;
        if (mi < 0 || mi >= M) mi = 0;
        Mesh& mm = parts[mi]; const Shader& sh = *shaders[mi];
        int a = r.foff[f], b = r.foff[f + 1];
        for (int i = a + 1; i + 1 < b; ++i) { auto v0 = emit(mm, sh, na[mi], a); auto v1 = emit(mm, sh, na[mi], i); auto v2 = emit(mm, sh, na[mi], i + 1); mm.tri.push_back({v0, v1, v2}); }
    }
    if (M == 1) return std::move(parts[0]);
    return merge_meshes(parts);  // world-space; the caller drops the assembled model to the plate once
}

}  // namespace printman
