#ifndef slic3r_PrintMan_Palette_hpp_
#define slic3r_PrintMan_Palette_hpp_

// Quantize a surface colour to the loaded filament palette: the PrintMan colour path evaluates a
// shader's linear-RGB Cout per refined face, then this maps it to the nearest loaded filament so the
// slicer can lay that face in that filament. The match metric is CIELAB DeltaE2000 (perceptual),
// reusing the fork's FlushPredict rather than a second colour converter.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "libslic3r/FlushVolPredictor.hpp"   // FlushPredict::RGBColor, calc_color_distance (DeltaE2000)
#include "libslic3r/PrintMan/Displace.hpp"   // V3

namespace Slic3r { namespace PrintMan {

// Encode one linear-light channel [0,1] to an 8-bit sRGB byte (the sRGB OETF). FlushPredict::RGB2LAB
// treats its RGBColor as sRGB and gamma-decodes it, so a shader's LINEAR Cout must be encoded here
// first, or the DeltaE match is computed in the wrong space.
inline unsigned char linear_to_srgb8(double c)
{
    // Map <=0 and NaN to 0 and clamp the top, so a NaN or out-of-gamut shader colour can't reach
    // lround as a non-finite value (undefined). Valid [0,1] inputs pass through unchanged.
    c = (c > 0.0) ? std::min(c, 1.0) : 0.0;
    const double s = (c <= 0.0031308) ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return (unsigned char) std::lround(s * 255.0);
}

// A linear-RGB surface colour as an sRGB-byte RGBColor, ready for FlushPredict's DeltaE2000.
inline FlushPredict::RGBColor srgb_from_linear(const V3 &linear_rgb)
{
    return FlushPredict::RGBColor(linear_to_srgb8(linear_rgb[0]),
                                  linear_to_srgb8(linear_rgb[1]),
                                  linear_to_srgb8(linear_rgb[2]));
}

// Nearest loaded filament to a linear-RGB surface colour, by CIELAB DeltaE2000. `palette` holds the
// loaded filament colours as sRGB bytes (filament_colour, decoded); index k is extruder k+1. Returns
// the 0-based palette index, or -1 for an empty palette. Ties resolve to the lower index (stable).
inline int nearest_filament(const V3 &linear_rgb, const std::vector<FlushPredict::RGBColor> &palette)
{
    if (palette.empty())
        return -1;
    const FlushPredict::RGBColor target = srgb_from_linear(linear_rgb);
    int   best   = 0;
    float best_d = FlushPredict::calc_color_distance(target, palette[0]);
    for (size_t k = 1; k < palette.size(); ++k) {
        const float d = FlushPredict::calc_color_distance(target, palette[k]);
        if (d < best_d) { best_d = d; best = int(k); }
    }
    return best;
}

// Decode one 8-bit sRGB channel to linear light (inverse of linear_to_srgb8).
inline double srgb8_to_linear(unsigned char v)
{
    const double s = double(v) / 255.0;
    return (s <= 0.04045) ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
}

inline V3 linear_from_srgb(const FlushPredict::RGBColor &c)
{
    return V3{{srgb8_to_linear(c.r), srgb8_to_linear(c.g), srgb8_to_linear(c.b)}};
}

// A deterministic spatial hash of a world point, quantized to `cell` mm, in [0,1). Two points more than
// a cell apart get uncorrelated values, so it drives a (white-noise) ordered dither over the surface.
inline double dither_hash(const V3 &p, double cell)
{
    if (! (cell > 0.0))
        cell = 0.5;   // a non-positive (or NaN) cell would divide to inf/NaN, then floor is undefined
    const long cx = long(std::floor(p[0] / cell)), cy = long(std::floor(p[1] / cell)), cz = long(std::floor(p[2] / cell));
    uint32_t   h  = uint32_t(cx) * 73856093u ^ uint32_t(cy) * 19349663u ^ uint32_t(cz) * 83492791u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return double((h ^ (h >> 16)) & 0xFFFFFFu) / double(0x1000000u);
}

// Dither a linear-RGB surface colour across the WHOLE loaded palette so a spatial mix reproduces any colour
// inside the palette's gamut. A printed surface tiles opaque filament patches and the eye area-averages them,
// so the reproduced colour is the linear-light convex combination sum(w_i * filament_i): the reachable set is
// the convex hull of the loaded colours, and the best match is T's projection onto that hull. We solve for
// those coverage weights w_i (>=0, sum 1) that minimise ||sum w_i P_i - T||^2 by Frank-Wolfe (each step moves
// mass toward the filament that most reduces the error -- converges into the hull, weights stay a valid
// distribution), then pick this cell's filament by sampling that distribution with the spatial dither value
// d01 in [0,1). Over the surface the areas land in the w_i proportions, so the average is the matched colour.
// (The old code mixed only the two nearest -- reachable colours were limited to hull EDGES, not its interior.)
inline int dither_filament(const V3 &linear_rgb, const std::vector<FlushPredict::RGBColor> &palette, double d01)
{
    const int n = int(palette.size());
    if (n == 0) return -1;
    if (n == 1) return 0;

    std::vector<V3> P(n);
    for (int i = 0; i < n; ++i) P[i] = linear_from_srgb(palette[i]);

    // Frank-Wolfe over the probability simplex, initialised at the perceptually nearest filament.
    std::vector<double> w(n, 0.0);
    const int init = nearest_filament(linear_rgb, palette);
    w[init] = 1.0;
    V3 cur = P[init];   // cur == sum_i w_i P_i, maintained incrementally
    for (int it = 0; it < 48; ++it) {
        // gradient of ||cur - T||^2 in w_s is 2 (cur - T).P_s; move toward the filament that minimises it.
        const V3 r{{cur[0] - linear_rgb[0], cur[1] - linear_rgb[1], cur[2] - linear_rgb[2]}};
        int    s = 0;
        double best = r[0] * P[0][0] + r[1] * P[0][1] + r[2] * P[0][2];
        for (int i = 1; i < n; ++i) {
            const double gi = r[0] * P[i][0] + r[1] * P[i][1] + r[2] * P[i][2];
            if (gi < best) { best = gi; s = i; }
        }
        const double g = 2.0 / (it + 2.0);   // standard FW step size
        for (int i = 0; i < n; ++i) w[i] *= (1.0 - g);
        w[s] += g;
        cur[0] = (1.0 - g) * cur[0] + g * P[s][0];
        cur[1] = (1.0 - g) * cur[1] + g * P[s][1];
        cur[2] = (1.0 - g) * cur[2] + g * P[s][2];
    }

    // Sample the coverage distribution: this cell takes the filament whose weight interval contains d01.
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { acc += w[i]; if (d01 < acc) return i; }
    return n - 1;
}

}}  // namespace Slic3r::PrintMan

#endif  // slic3r_PrintMan_Palette_hpp_
