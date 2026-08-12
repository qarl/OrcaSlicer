#ifndef slic3r_PrintMan_Displace_hpp_
#define slic3r_PrintMan_Displace_hpp_

// Displacement shading for the amplification engine: a shader moves each refined surface
// vertex along its normal. Zero-dependency std (like USDSubdiv.hpp), ported from and
// validated bit-for-bit against printman/shade.py (Device/Gridwork/Noise) and mesh.py
// (area-weighted vertex_normals). See the PrintMan design docs for the model and the vertex rule.
// NOTE: this uses single per-band vertex normals; across band seams that is only continuous while
// the displacement magnitude is >~ the layer height (see slice_cage_placement). Band-invariant
// normals (for low-amplitude fields) are a follow-up.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "libslic3r/TriangleMesh.hpp"  // indexed_triangle_set

namespace Slic3r { namespace PrintMan {

using V3 = std::array<double, 3>;

// A shader is (world point, unit normal) -> displacement in mm, +out along the normal.
using DisplaceShader = std::function<double(const V3 &point, const V3 &normal)>;

// ---- ported displacement shaders (printman/shade.py) --------------------------------------

struct Device {
    double layer_height = 0.2;
    double extrusion_width = 0.42;
    double self_supporting_angle() const { return std::atan2(extrusion_width, layer_height); }
    double beads(double n) const { return n * extrusion_width; }
};

// 3D lattice gridwork, default params (shade.py Gridwork).
struct Gridwork {
    double depth_beads = 3.0;
    double pitch = 1.5;
    double groove = 0.75;  // pitch/2
    double phase = 0.75;   // pitch/2
    bool   raised = true;
    bool   printable = true;
    static constexpr double kRetireFrom = 0.9;

    double rib() const { return pitch - groove; }
    double max_depth(const Device &d) const { return 0.5 * groove * std::tan(d.self_supporting_angle()); }
    double effective_depth(const Device &d) const {
        double asked = d.beads(depth_beads);
        return printable ? std::min(asked, max_depth(d)) : asked;
    }
    double operator()(const V3 &p, const V3 &n, const Device &d) const {
        const double depth = effective_depth(d), angle = d.self_supporting_angle(), half_rib = rib() / 2.0;
        double per_axis[3];
        for (int k = 0; k < 3; ++k) {
            const double t = (p[k] - phase) / pitch;
            const double dist = std::min(t - std::floor(t), 1.0 - (t - std::floor(t))) * pitch;
            per_axis[k] = (k == 2 && printable) ? std::clamp((dist - half_rib) * std::tan(angle), 0.0, depth)
                                                : (dist > half_rib ? depth : 0.0);
        }
        for (int k = 0; k < 3; ++k) {
            const double f = std::clamp((std::abs(n[k]) - kRetireFrom) / (1.0 - kRetireFrom), 0.0, 1.0);
            const double retire = f * f * (3.0 - 2.0 * f);
            per_axis[k] = per_axis[k] * (1.0 - retire) + depth * retire;
        }
        const double cut = std::min({per_axis[0], per_axis[1], per_axis[2]});
        return raised ? (depth - cut) : -cut;
    }
};

// Value-noise displacement, default params (shade.py Noise). Bit-for-bit vs shade.py.
struct Noise {
    double amplitude_beads = 1.0;
    double scale = 4.0;
    long   seed = 0;
    static double hash_lattice(long cx, long cy, long cz, long seed) {
        uint32_t h = (uint32_t)(cx * 73856093) ^ (uint32_t)(cy * 19349663) ^ (uint32_t)(cz * 83492791);
        h = h ^ (uint32_t)seed;
        h = (h ^ (h >> 13)) * 1274126177u;
        return double((h ^ (h >> 16)) & 0xFFFFFFu) / double(0xFFFFFFu);
    }
    double operator()(const V3 &point, const V3 & /*normal*/, const Device &d) const {
        long cell[3]; double smooth[3];
        for (int k = 0; k < 3; ++k) {
            const double p = point[k] / scale, fl = std::floor(p);
            cell[k] = long(fl);
            const double frac = p - fl;
            smooth[k] = frac * frac * (3.0 - 2.0 * frac);
        }
        double total = 0.0;
        for (int c = 0; c < 8; ++c) {
            const long off[3] = {(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1};
            double w = 1.0;
            for (int k = 0; k < 3; ++k) w *= (off[k] == 1) ? smooth[k] : (1.0 - smooth[k]);
            total += w * hash_lattice(cell[0] + off[0], cell[1] + off[1], cell[2] + off[2], seed);
        }
        return d.beads(amplitude_beads) * (total * 2.0 - 1.0);
    }
};

// ---- geometry: area-weighted vertex normals + displacement application ----------------------

// Area-weighted unit vertex normals of a world-space mesh (mesh.py vertex_normals), in double.
inline std::vector<V3> vertex_normals(const indexed_triangle_set &its) {
    std::vector<V3> N(its.vertices.size(), V3{{0, 0, 0}});
    for (const auto &t : its.indices) {
        const Vec3f &a = its.vertices[t[0]], &b = its.vertices[t[1]], &c = its.vertices[t[2]];
        const double e1x = double(b.x()) - a.x(), e1y = double(b.y()) - a.y(), e1z = double(b.z()) - a.z();
        const double e2x = double(c.x()) - a.x(), e2y = double(c.y()) - a.y(), e2z = double(c.z()) - a.z();
        const V3 w{{e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x}};
        for (int k = 0; k < 3; ++k) { N[t[k]][0] += w[0]; N[t[k]][1] += w[1]; N[t[k]][2] += w[2]; }
    }
    for (V3 &n : N) {
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 0.0) { n[0] /= len; n[1] /= len; n[2] /= len; }
    }
    return N;
}

// Displace each vertex of a world-space mesh along its (single, area-weighted) normal by
// shader(point, normal) mm. Returns max |displacement|. Matches the current shade.py pipeline
// (single fine-mesh vertex normals); the per-face-average rule is a later refinement.
inline double apply_displacement(indexed_triangle_set &its, const DisplaceShader &shader) {
    const std::vector<V3> N = vertex_normals(its);
    double maxd = 0.0;
    for (size_t i = 0; i < its.vertices.size(); ++i) {
        const Vec3f &vf = its.vertices[i];
        const V3 p{{double(vf.x()), double(vf.y()), double(vf.z())}};
        const double d = shader(p, N[i]);
        its.vertices[i] = Vec3f(float(p[0] + d * N[i][0]), float(p[1] + d * N[i][1]), float(p[2] + d * N[i][2]));
        maxd = std::max(maxd, std::abs(d));
    }
    return maxd;
}

}}  // namespace Slic3r::PrintMan

#endif  // slic3r_PrintMan_Displace_hpp_
