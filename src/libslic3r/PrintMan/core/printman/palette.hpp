#pragma once
//
// Colour -> filament classification (the CORE, host-neutral half of the colour path). A shader's
// linear-RGB Cout is matched to the nearest loaded filament by perceptual CIELAB DeltaE2000, or
// dithered across the whole palette so a spatial mix reproduces any in-gamut colour. This is pure
// scalar colour math -- no geometry, no 2D booleans -- so it lives in the neutral core; the sink owns
// where the classified colour is laid down (the wall-claim). Ported from the fork's Palette.hpp +
// FlushVolPredictor colour math; DeltaE2000 is validated against the Sharma et al. (2005) reference.
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "printman/geom.hpp"   // V3 = std::array<double,3>

namespace printman {

struct RGBColor { unsigned char r = 0, g = 0, b = 0;
    RGBColor() = default;
    RGBColor(unsigned char r_, unsigned char g_, unsigned char b_) : r(r_), g(g_), b(b_) {} };
struct LABColor { double l = 0, a = 0, b = 0;
    LABColor() = default;
    LABColor(double l_, double a_, double b_) : l(l_), a(a_), b(b_) {} };

// sRGB byte triple -> CIELAB (D65). The RGBColor is treated as sRGB and gamma-decoded.
inline LABColor rgb2lab(const RGBColor& c) {
    auto gamma = [](double x) { return x > 0.04045 ? std::pow((x + 0.055) / 1.055, 2.4) : x / 12.92; };
    double R = gamma(double(c.r) / 255.0) * 100.0, G = gamma(double(c.g) / 255.0) * 100.0, B = gamma(double(c.b) / 255.0) * 100.0;
    double x = 0.412453 * R + 0.357580 * G + 0.180423 * B;
    double y = 0.212671 * R + 0.715160 * G + 0.072169 * B;
    double z = 0.019334 * R + 0.119193 * G + 0.950227 * B;
    const double XN = 95.0489, YN = 100.0, ZN = 108.8840;
    auto f = [](double t) { return t > 0.008856 ? std::pow(t, 1.0 / 3.0) : 7.787 * t + 0.137931; };
    double xn = f(x / XN), yn = f(y / YN), zn = f(z / ZN);
    return LABColor(116.0 * yn - 16.0, 500.0 * (xn - yn), 200.0 * (yn - zn));
}

// CIEDE2000 colour difference between two Lab colours (Sharma et al. 2005). K_L=K_C=K_H=1.
inline double delta_e(const LABColor& lab1, const LABColor& lab2) {
    static const double pow_25_7 = std::pow(25.0, 7.0);
    auto d2r = [](double d) { return d * M_PI / 180.0; };
    const double C1 = std::sqrt(lab1.a * lab1.a + lab1.b * lab1.b);
    const double C2 = std::sqrt(lab2.a * lab2.a + lab2.b * lab2.b);
    const double Cm = (C1 + C2) / 2.0, Cm7 = std::pow(Cm, 7.0);
    const double G = 0.5 * (1.0 - std::sqrt(Cm7 / (Cm7 + pow_25_7)));
    const double a1 = (1.0 + G) * lab1.a, a2 = (1.0 + G) * lab2.a;
    const double c1 = std::sqrt(a1 * a1 + lab1.b * lab1.b), c2 = std::sqrt(a2 * a2 + lab2.b * lab2.b);
    auto hue = [](double a, double b) { if (a == 0 && b == 0) return 0.0; double h = std::atan2(b, a); return h < 0 ? h + 2 * M_PI : h; };
    const double h1 = hue(a1, lab1.b), h2 = hue(a2, lab2.b);
    const double dL = lab2.l - lab1.l, dC = c2 - c1;
    double dH;
    if (c1 * c2 == 0) dH = 0;
    else { double dh = h2 - h1; if (dh < -M_PI) dh += 2 * M_PI; else if (dh > M_PI) dh -= 2 * M_PI; dH = 2 * std::sqrt(c1 * c2) * std::sin(dh / 2.0); }
    const double Lm = (lab1.l + lab2.l) / 2.0, Cmean = (c1 + c2) / 2.0;
    double Hm, hs = h1 + h2;
    if (c1 * c2 == 0) Hm = hs;
    else if (std::fabs(h1 - h2) <= M_PI) Hm = hs / 2.0;
    else Hm = (hs < 2 * M_PI) ? (hs + 2 * M_PI) / 2.0 : (hs - 2 * M_PI) / 2.0;
    const double T = 1 - 0.17 * std::cos(Hm - d2r(30)) + 0.24 * std::cos(2 * Hm) + 0.32 * std::cos(3 * Hm + d2r(6)) - 0.20 * std::cos(4 * Hm - d2r(63));
    const double dth = d2r(30) * std::exp(-std::pow((Hm - d2r(275)) / d2r(25), 2.0));
    const double Cm7b = std::pow(Cmean, 7.0);
    const double RC = 2.0 * std::sqrt(Cm7b / (Cm7b + pow_25_7));
    const double L2 = std::pow(Lm - 50.0, 2.0);
    const double SL = 1 + (0.015 * L2) / std::sqrt(20.0 + L2);
    const double SC = 1 + 0.045 * Cmean, SH = 1 + 0.015 * Cmean * T, RT = -std::sin(2 * dth) * RC;
    return std::sqrt(dL / SL * (dL / SL) + dC / SC * (dC / SC) + dH / SH * (dH / SH) + RT * (dC / SC) * (dH / SH));
}
inline double delta_e(const RGBColor& a, const RGBColor& b) { return delta_e(rgb2lab(a), rgb2lab(b)); }

// linear light <-> sRGB byte (the sRGB OETF and its inverse). Clamps/rejects non-finite input.
inline unsigned char linear_to_srgb8(double c) {
    c = (c > 0.0) ? std::min(c, 1.0) : 0.0;
    const double s = (c <= 0.0031308) ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    return (unsigned char)std::lround(s * 255.0);
}
inline double srgb8_to_linear(unsigned char v) {
    const double s = double(v) / 255.0;
    return (s <= 0.04045) ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
}
inline RGBColor srgb_from_linear(const V3& lin) { return RGBColor(linear_to_srgb8(lin[0]), linear_to_srgb8(lin[1]), linear_to_srgb8(lin[2])); }
inline V3 linear_from_srgb(const RGBColor& c) { return V3{{srgb8_to_linear(c.r), srgb8_to_linear(c.g), srgb8_to_linear(c.b)}}; }

// Nearest loaded filament (0-based index into palette, sRGB bytes) to a linear-RGB colour, by
// DeltaE2000. Ties resolve to the lower index. -1 for an empty palette.
inline int nearest_filament(const V3& lin, const std::vector<RGBColor>& palette) {
    if (palette.empty()) return -1;
    const LABColor t = rgb2lab(srgb_from_linear(lin));
    int best = 0; double bd = delta_e(t, rgb2lab(palette[0]));
    for (size_t k = 1; k < palette.size(); ++k) { double d = delta_e(t, rgb2lab(palette[k])); if (d < bd) { bd = d; best = int(k); } }
    return best;
}

// Deterministic spatial hash of a world point, quantized to `cell` mm, in [0,1) -- drives the dither.
inline double dither_hash(const V3& p, double cell) {
    if (!(cell > 0.0)) cell = 0.5;
    const long cx = long(std::floor(p[0] / cell)), cy = long(std::floor(p[1] / cell)), cz = long(std::floor(p[2] / cell));
    std::uint32_t h = std::uint32_t(cx) * 73856093u ^ std::uint32_t(cy) * 19349663u ^ std::uint32_t(cz) * 83492791u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return double((h ^ (h >> 16)) & 0xFFFFFFu) / double(0x1000000u);
}

// Dither a linear-RGB colour across the whole palette: a printed surface tiles opaque filament
// patches and the eye area-averages them, so the reproduced colour is the linear convex combination
// of the loaded colours. Solve the coverage weights minimising ||sum w_i P_i - T||^2 by Frank-Wolfe
// (over the simplex), then this cell takes the filament whose weight interval contains d01.
inline int dither_filament(const V3& lin, const std::vector<RGBColor>& palette, double d01) {
    const int n = int(palette.size());
    if (n == 0) return -1;
    if (n == 1) return 0;
    std::vector<V3> P(n);
    for (int i = 0; i < n; ++i) P[i] = linear_from_srgb(palette[i]);
    std::vector<double> w(n, 0.0);
    const int init = nearest_filament(lin, palette);
    w[init] = 1.0;
    V3 cur = P[init];
    for (int it = 0; it < 48; ++it) {
        const V3 r{{cur[0] - lin[0], cur[1] - lin[1], cur[2] - lin[2]}};
        int s = 0; double best = r[0] * P[0][0] + r[1] * P[0][1] + r[2] * P[0][2];
        for (int i = 1; i < n; ++i) { double gi = r[0] * P[i][0] + r[1] * P[i][1] + r[2] * P[i][2]; if (gi < best) { best = gi; s = i; } }
        const double g = 2.0 / (it + 2.0);
        for (int i = 0; i < n; ++i) w[i] *= (1.0 - g);
        w[s] += g;
        cur[0] = (1 - g) * cur[0] + g * P[s][0]; cur[1] = (1 - g) * cur[1] + g * P[s][1]; cur[2] = (1 - g) * cur[2] + g * P[s][2];
    }
    double acc = 0.0;
    for (int i = 0; i < n; ++i) { acc += w[i]; if (d01 < acc) return i; }
    return n - 1;
}

// Overwrite each vertex's Cout AOV with its nearest-filament colour (linear) -- a filament-quantized
// preview of the surface colour (what an FDM print would reproduce with this palette). cout_k is the
// Cout AOV index; dither spreads the choice spatially so the eye area-averages toward the true colour.
inline void quantize_cout(Mesh& m, const std::vector<RGBColor>& palette, int cout_k,
                          bool dither = false, double cell = 0.5) {
    if (palette.empty() || cout_k < 0 || cout_k >= (int)m.aov.size()) return;
    auto& A = m.aov[cout_k];
    for (size_t v = 0; v < A.size(); ++v) {
        V3 c{{A[v][0], A[v][1], A[v][2]}};
        int f = dither ? dither_filament(c, palette, dither_hash(V3{{m.pos[v][0], m.pos[v][1], m.pos[v][2]}}, cell))
                       : nearest_filament(c, palette);
        V3 lin = linear_from_srgb(palette[f]);
        A[v] = {{(float)lin[0], (float)lin[1], (float)lin[2]}};
    }
}

}  // namespace printman
