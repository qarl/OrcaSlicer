#ifndef slic3r_PrintMan_Displace_hpp_
#define slic3r_PrintMan_Displace_hpp_

// Displacement shading for the amplification engine: a shader moves each refined surface
// vertex along its normal. Zero-dependency std (like USDSubdiv.hpp), ported from and
// validated bit-for-bit against printman/shade.py (Device/Gridwork/Noise) and mesh.py
// (mesh.py). See the PrintMan design docs for the model and the per-face-average vertex rule.
// NOTE: normals are computed per-band (on each band's core mesh), so across band seams continuity
// only holds while the displacement magnitude is >~ the layer height (see slice_cage_placement);
// band-invariant normals for low-amplitude fields are a deferred follow-up.

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

// ---- geometry: the per-face-average primvar reducer + displacement application --------------

// Unit face normal of a triangle read as double. Zero if degenerate.
inline V3 face_normal(const Vec3f &a, const Vec3f &b, const Vec3f &c) {
    const double e1x = double(b.x()) - a.x(), e1y = double(b.y()) - a.y(), e1z = double(b.z()) - a.z();
    const double e2x = double(c.x()) - a.x(), e2y = double(c.y()) - a.y(), e2z = double(c.z()) - a.z();
    const V3 w{{e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x}};
    const double len = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    return len > 0.0 ? V3{{w[0] / len, w[1] / len, w[2] / len}} : V3{{0, 0, 0}};
}

// Per-face-average primvar reducer: for each triangle, `contrib(vertex_pos, unit_face_normal)`
// yields a Vec3; accumulate onto each of the triangle's vertices, then divide by the
// incident-face count. One averaged Vec3 per vertex. Generic so it serves displacement (a d*n
// vector) now and colour later. Validated against tools/gen_reduce_golden.py.
template <typename Contrib>
inline std::vector<V3> per_face_average(const indexed_triangle_set &its, const Contrib &contrib) {
    std::vector<V3>  accum(its.vertices.size(), V3{{0, 0, 0}});
    std::vector<int> count(its.vertices.size(), 0);
    for (const auto &t : its.indices) {
        const V3 fn = face_normal(its.vertices[t[0]], its.vertices[t[1]], its.vertices[t[2]]);
        for (int k = 0; k < 3; ++k) {
            const Vec3f &vk = its.vertices[t[k]];
            const V3 c = contrib(V3{{double(vk.x()), double(vk.y()), double(vk.z())}}, fn);
            accum[t[k]][0] += c[0]; accum[t[k]][1] += c[1]; accum[t[k]][2] += c[2];
            ++count[t[k]];
        }
    }
    for (size_t i = 0; i < accum.size(); ++i)
        if (count[i] > 0) { accum[i][0] /= count[i]; accum[i][1] /= count[i]; accum[i][2] /= count[i]; }
    return accum;
}

// Displace each vertex by the per-face-average of d(point, face_normal) * face_normal (mm):
// watertight (one position per shared vertex, no crack), faithful to normal-dependent shaders
// (each face's real orientation), and auto-damping at edges (the averaged vector shrinks where
// the incident normals diverge). Returns max |displacement|.
inline double apply_displacement(indexed_triangle_set &its, const DisplaceShader &shader) {
    const std::vector<V3> dv = per_face_average(its, [&](const V3 &p, const V3 &n) {
        const double d = shader(p, n);
        return V3{{d * n[0], d * n[1], d * n[2]}};
    });
    double maxd = 0.0;
    for (size_t i = 0; i < its.vertices.size(); ++i) {
        Vec3f &v = its.vertices[i];
        v = Vec3f(float(double(v.x()) + dv[i][0]), float(double(v.y()) + dv[i][1]), float(double(v.z()) + dv[i][2]));
        const double mag = std::sqrt(dv[i][0] * dv[i][0] + dv[i][1] * dv[i][1] + dv[i][2] * dv[i][2]);
        if (mag > maxd) maxd = mag;
    }
    return maxd;
}

}}  // namespace Slic3r::PrintMan

#endif  // slic3r_PrintMan_Displace_hpp_
