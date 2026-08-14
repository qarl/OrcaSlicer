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

// Dither a linear-RGB surface colour to the loaded filaments so a spatial mix reproduces intermediate
// colours the palette can't hit outright. Picks the two nearest filaments by DeltaE2000, projects the
// target onto their segment in LINEAR light for the area fraction toward the second, and chooses one
// vs. the other by the dither value `d01` in [0,1). Returns a 0-based palette index, or -1 if empty.
inline int dither_filament(const V3 &linear_rgb, const std::vector<FlushPredict::RGBColor> &palette, double d01)
{
    if (palette.empty())
        return -1;
    if (palette.size() == 1)
        return 0;
    const FlushPredict::RGBColor target = srgb_from_linear(linear_rgb);
    int   k0 = 0, k1 = -1;
    float d0 = FlushPredict::calc_color_distance(target, palette[0]), d1 = 1e30f;
    for (size_t k = 1; k < palette.size(); ++k) {
        const float d = FlushPredict::calc_color_distance(target, palette[k]);
        if (d < d0)      { d1 = d0; k1 = k0; d0 = d; k0 = int(k); }
        else if (d < d1) { d1 = d;  k1 = int(k); }
    }
    if (k1 < 0)
        return k0;
    const V3     a  = linear_from_srgb(palette[k0]), b = linear_from_srgb(palette[k1]);
    const V3     ab{{b[0] - a[0], b[1] - a[1], b[2] - a[2]}};
    const double dd = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    double       f  = (dd > 1e-12) ? ((linear_rgb[0] - a[0]) * ab[0] + (linear_rgb[1] - a[1]) * ab[1] + (linear_rgb[2] - a[2]) * ab[2]) / dd : 0.0;
    f = std::clamp(f, 0.0, 1.0);
    return (d01 < f) ? k1 : k0;
}

}}  // namespace Slic3r::PrintMan

#endif  // slic3r_PrintMan_Palette_hpp_
