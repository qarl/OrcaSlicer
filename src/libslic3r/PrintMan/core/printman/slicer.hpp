#pragma once
//
// The core's own triangle -> Z-plane slicer. No libslic3r, no external clipper. Each triangle
// crossing a layer plane contributes one oriented segment, carrying every AOV interpolated to the
// two crossing points. Segments follow the triangle winding order so a downstream NON-ZERO
// winding fill treats self-intersecting folds as solid union, not floating islands.
//
#include <algorithm>
#include <array>
#include <vector>

#include "printman/geom.hpp"
#include "printman/palette.hpp"

namespace printman {

struct Seg {
    V2 a, b;                 // endpoints in the XY plane (mm)
    std::vector<V3> va, vb;  // per-AOV values at a, b (same order as Mesh::aov_names)
    int filament = -1;       // classified filament index (classify_layers), -1 = unclassified
};

using LayerSegs = std::vector<Seg>;

// Classify each segment to a filament from its midpoint Cout (linear RGB), by DeltaE2000 nearest, or
// dithered across the palette (spatial hash of the segment midpoint at the layer z). This is the
// CORE, host-neutral colour decision -- "which filament here"; the sink owns "where to lay it down"
// (the wall-claim). cout_k is the Cout AOV index; palette is the loaded filament colours (sRGB bytes).
inline void classify_layers(std::vector<LayerSegs>& layers, const std::vector<double>& zs,
                            const std::vector<RGBColor>& palette, int cout_k,
                            bool dither = false, double dither_cell = 0.5) {
    if (palette.empty() || cout_k < 0) return;
    for (size_t li = 0; li < layers.size(); ++li) {
        const double z = li < zs.size() ? zs[li] : 0.0;
        for (auto& s : layers[li]) {
            if (cout_k >= (int)s.va.size() || cout_k >= (int)s.vb.size()) continue;
            const V3& ca = s.va[cout_k]; const V3& cb = s.vb[cout_k];
            V3 c{{(ca[0]+cb[0])*0.5, (ca[1]+cb[1])*0.5, (ca[2]+cb[2])*0.5}};
            s.filament = dither
                ? dither_filament(c, palette, dither_hash(V3{{(s.a[0]+s.b[0])*0.5, (s.a[1]+s.b[1])*0.5, z}}, dither_cell))
                : nearest_filament(c, palette);
        }
    }
}

inline std::vector<LayerSegs> slice_mesh(const Mesh& m, const std::vector<double>& zs) {
    const int na = int(m.aov.size());
    std::vector<LayerSegs> out(zs.size());
    for (const auto& t : m.tri) {
        const std::array<float, 3>* vp[3] = {&m.pos[t[0]], &m.pos[t[1]], &m.pos[t[2]]};
        double tzmin = std::min({(*vp[0])[2], (*vp[1])[2], (*vp[2])[2]});
        double tzmax = std::max({(*vp[0])[2], (*vp[1])[2], (*vp[2])[2]});
        for (size_t li = 0; li < zs.size(); ++li) {
            double Z = zs[li];
            if (Z < tzmin || Z > tzmax) continue;
            V2 pt[2]; std::vector<V3> col[2]; int nc = 0;
            for (int e = 0; e < 3 && nc < 2; ++e) {
                const auto& u = *vp[e]; const auto& v = *vp[(e + 1) % 3];
                double zu = u[2], zv = v[2];
                if (!((zu <= Z && zv > Z) || (zv <= Z && zu > Z))) continue;
                double tt = (Z - zu) / (zv - zu);
                pt[nc] = {{u[0] + (v[0] - u[0]) * tt, u[1] + (v[1] - u[1]) * tt}};
                col[nc].resize(na);
                for (int k = 0; k < na; ++k) {
                    const auto& cu = m.aov[k][t[e]]; const auto& cv = m.aov[k][t[(e + 1) % 3]];
                    col[nc][k] = {{cu[0] + (cv[0] - cu[0]) * (float)tt,
                                   cu[1] + (cv[1] - cu[1]) * (float)tt,
                                   cu[2] + (cv[2] - cu[2]) * (float)tt}};
                }
                ++nc;
            }
            if (nc == 2) out[li].push_back(Seg{pt[0], pt[1], std::move(col[0]), std::move(col[1])});
        }
    }
    return out;
}

}  // namespace printman
