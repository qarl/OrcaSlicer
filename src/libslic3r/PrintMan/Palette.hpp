#ifndef slic3r_PrintMan_Palette_hpp_
#define slic3r_PrintMan_Palette_hpp_

// Quantize a surface colour to the loaded filament palette: the PrintMan colour path evaluates a
// shader's linear-RGB Cout per refined face, then this maps it to the nearest loaded filament so the
// slicer can lay that face in that filament. The match metric is CIELAB DeltaE2000 (perceptual),
// reusing the fork's FlushPredict rather than a second colour converter.

#include <algorithm>
#include <cmath>
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

}}  // namespace Slic3r::PrintMan

#endif  // slic3r_PrintMan_Palette_hpp_
