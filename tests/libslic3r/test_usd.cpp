#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>

#include <catch2/catch_all.hpp>

#include <Eigen/Dense>

#include "libslic3r/Model.hpp"
#include "libslic3r/PrintConfig.hpp"
#include "libslic3r/Format/USD.hpp"
#include "libslic3r/Format/USDSubdiv.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/Layer.hpp"
#include "libslic3r/ModelArrange.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/Exception.hpp"
#include "libslic3r/PrintMan/Engine.hpp"
#include "libslic3r/PrintMan/Palette.hpp"
#include "libslic3r/TriangleMeshSlicer.hpp"

using namespace Slic3r;

static inline std::string usd_path(const char *path)
{
    return std::string(TEST_DATA_DIR) + "/test_usd/" + path;
}

// resources_dir() is process-global; restore it (this test binary is shared).
struct UsdResourcesFixture
{
    UsdResourcesFixture() : previous(resources_dir()) { set_resources_dir(USD_PLUGIN_PARENT_DIR); }
    ~UsdResourcesFixture() { set_resources_dir(previous); }

    std::string previous;
};

// Palette match: the colour path evaluates a shader's linear-RGB Cout per refined face, then maps it
// to the nearest loaded filament by CIELAB DeltaE2000 (reusing FlushPredict). Pure -- no USD, no OSL,
// no fixture. The shader's Cout is LINEAR, so the encode into sRGB (what DeltaE2000 expects) is part
// of the contract and is checked here too.
SCENARIO("PrintMan quantizes a surface colour to the nearest filament", "[PrintMan][palette][color]")
{
    using FlushPredict::RGBColor;
    GIVEN("an R/G/B/black/white filament palette (sRGB bytes)") {
        const std::vector<RGBColor> palette = {
            RGBColor(255,   0,   0),   // 0: red
            RGBColor(  0, 255,   0),   // 1: green
            RGBColor(  0,   0, 255),   // 2: blue
            RGBColor(  0,   0,   0),   // 3: black
            RGBColor(255, 255, 255),   // 4: white
        };
        THEN("a pure linear primary maps to its own filament") {
            REQUIRE(PrintMan::nearest_filament(PrintMan::V3{{1, 0, 0}}, palette) == 0);
            REQUIRE(PrintMan::nearest_filament(PrintMan::V3{{0, 1, 0}}, palette) == 1);
            REQUIRE(PrintMan::nearest_filament(PrintMan::V3{{0, 0, 1}}, palette) == 2);
            REQUIRE(PrintMan::nearest_filament(PrintMan::V3{{0, 0, 0}}, palette) == 3);
            REQUIRE(PrintMan::nearest_filament(PrintMan::V3{{1, 1, 1}}, palette) == 4);
        }
        THEN("an off-primary maps to the perceptually nearest filament") {
            // A strong but impure red is still nearest red, not black/white, under DeltaE2000.
            REQUIRE(PrintMan::nearest_filament(PrintMan::V3{{0.6, 0.06, 0.06}}, palette) == 0);
            // Mid grey is nearest an achromatic filament (black or white), never a chroma primary.
            const int g = PrintMan::nearest_filament(PrintMan::V3{{0.5, 0.5, 0.5}}, palette);
            REQUIRE((g == 3 || g == 4));
        }
    }
    GIVEN("an empty palette") {
        THEN("there is no filament to choose") {
            REQUIRE(PrintMan::nearest_filament(PrintMan::V3{{0.2, 0.4, 0.6}}, {}) == -1);
        }
    }
    GIVEN("the linear->sRGB encode") {
        THEN("endpoints hit the byte extremes and 0.5 linear lands near sRGB 188") {
            REQUIRE(int(PrintMan::linear_to_srgb8(0.0)) == 0);
            REQUIRE(int(PrintMan::linear_to_srgb8(1.0)) == 255);
            const int mid = int(PrintMan::linear_to_srgb8(0.5));
            REQUIRE(mid >= 186);
            REQUIRE(mid <= 190);
        }
        THEN("non-finite and out-of-gamut inputs clamp into range, never garbage") {
            REQUIRE(int(PrintMan::linear_to_srgb8(std::nan(""))) == 0);   // NaN shader colour -> black, not UB
            REQUIRE(int(PrintMan::linear_to_srgb8(-1.0))         == 0);
            REQUIRE(int(PrintMan::linear_to_srgb8(2.0))          == 255);
        }
    }
}

// Amplification colour (Phase B, engine): slice a cage with a ColorField whose Cout is red below the
// mid-Z and blue above. The engine deposits a colour ribbon along each refined face's slice segment and
// hands back [layer][channel] wall bands (clipped to the layer contour). Assert the split follows the
// shader in Z (bottom third carries red = channel 0, top third blue = channel 1) and that each channel
// is a band WITHIN the layer, not the whole layer. Uses a C++ colour lambda -- no OSL, always built.
SCENARIO_METHOD(UsdResourcesFixture, "The engine splits an amplified scene into per-filament layers", "[usd][subdiv][PrintMan][color]")
{
    GIVEN("a subdivision cube coloured red (low Z) / blue (high Z) over a 2-filament palette") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, true));
        const ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());

        const Transform3d m = vol->get_matrix();
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const Vec3f &v : vol->mesh().its.vertices) {
            const float z = float((m * v.cast<double>()).z());
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
            zs.push_back(z);
        REQUIRE(zs.size() > 10);

        MeshSlicingParamsEx params;
        params.trafo      = m;
        params.subdiv_tol = 0.05;

        const double zmid = 0.5 * (double(zs.front()) + zs.back());
        PrintMan::ColorField color;
        color.palette = {FlushPredict::RGBColor(255, 0, 0), FlushPredict::RGBColor(0, 0, 255)};
        color.eval    = [zmid](const PrintMan::V3 &p, const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &) {
            return p[2] < zmid ? PrintMan::V3{{1, 0, 0}} : PrintMan::V3{{0, 0, 1}};
        };

        std::vector<std::vector<ExPolygons>> seg;
        const std::vector<ExPolygons>        layers =
            PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, {}, color, &seg);

        REQUIRE(layers.size() == zs.size());
        REQUIRE(seg.size()    == zs.size());

        auto area = [](const ExPolygons &e) { double a = 0.0; for (const ExPolygon &p : e) a += p.area(); return a; };

        THEN("layers carry a within-layer wall band, and the split follows the shader in Z") {
            const double lo3 = double(zs.front()) + (double(zs.back()) - zs.front()) / 3.0;
            const double hi3 = double(zs.back())  - (double(zs.back()) - zs.front()) / 3.0;
            size_t red_layers = 0, blue_layers = 0;
            for (size_t L = 0; L < zs.size(); ++ L) {
                REQUIRE(seg[L].size() == 2);
                const double ared = area(seg[L][0]), ablue = area(seg[L][1]);
                const bool   has_red = ared > 0.0, has_blue = ablue > 0.0;
                if (layers[L].empty()) { REQUIRE_FALSE(has_red); REQUIRE_FALSE(has_blue); continue; }
                REQUIRE((has_red || has_blue));                       // the layer surface is coloured
                REQUIRE(ared  <= area(layers[L]) + 1e-6);             // each channel clipped to the layer
                REQUIRE(ablue <= area(layers[L]) + 1e-6);
                if (has_red)  ++ red_layers;
                if (has_blue) ++ blue_layers;
                if (zs[L] < lo3) { REQUIRE(has_red);  REQUIRE(ared  < area(layers[L])); }  // a band, not the whole layer
                if (zs[L] > hi3) { REQUIRE(has_blue); REQUIRE(ablue < area(layers[L])); }
            }
            REQUIRE(red_layers  > 0);
            REQUIRE(blue_layers > 0);
        }
    }
}

// Dithering (engine): a constant INTERMEDIATE colour (the linear midpoint of the two filaments) is
// unreachable by either filament, so hard-quantize picks ONE filament for the whole surface. Dithering
// instead spreads both filaments in a spatial pattern that area-averages to the target. Assert
// nearest -> exactly one channel, dither -> both channels present and roughly balanced. C++ lambda, no OSL.
SCENARIO_METHOD(UsdResourcesFixture, "The engine dithers an intermediate colour across filaments", "[usd][subdiv][PrintMan][color][dither]")
{
    GIVEN("a cube shaded a constant purple over a red/blue 2-filament palette") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, true));
        const ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());

        const Transform3d m = vol->get_matrix();
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const Vec3f &v : vol->mesh().its.vertices) {
            const float z = float((m * v.cast<double>()).z());
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
            zs.push_back(z);
        REQUIRE(zs.size() > 10);
        MeshSlicingParamsEx params;
        params.trafo      = m;
        params.subdiv_tol = 0.05;

        PrintMan::ColorField color;
        color.palette     = {FlushPredict::RGBColor(255, 0, 0), FlushPredict::RGBColor(0, 0, 255)};
        color.band_width  = 0.5;   // match the dither cell so the pattern is distinct, not smeared
        color.dither_cell = 0.5;
        color.eval        = [](const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &) {
            return PrintMan::V3{{0.5, 0.0, 0.5}};   // linear midpoint of red & blue -- unreachable by one filament
        };

        auto total = [](const std::vector<std::vector<ExPolygons>> &seg, size_t k) {
            double a = 0.0;
            for (const std::vector<ExPolygons> &layer : seg)
                if (k < layer.size())
                    for (const ExPolygon &p : layer[k])
                        a += p.area();
            return a;
        };

        WHEN("hard-quantized (no dither)") {
            std::vector<std::vector<ExPolygons>> seg;
            PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, {}, color, &seg);
            THEN("exactly one filament wins the whole surface") {
                REQUIRE((total(seg, 0) == 0.0) != (total(seg, 1) == 0.0));
            }
        }
        WHEN("dithered") {
            color.dither = true;
            std::vector<std::vector<ExPolygons>> seg;
            PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, {}, color, &seg);
            const double a0 = total(seg, 0), a1 = total(seg, 1);
            THEN("both filaments appear and are roughly balanced (the mix averages to the target)") {
                REQUIRE(a0 > 0.0);
                REQUIRE(a1 > 0.0);
                REQUIRE(std::abs(a0 - a1) < 0.5 * (a0 + a1));   // neither dominates; ~50/50 for f=0.5
            }
        }
        WHEN("three filaments are loaded and the target sits between two of them") {
            // Yellow (linear midpoint of red & green) over a red/green/blue palette: the two nearest are
            // red & green, so the dither must mix those and never reach for blue -- the N>2 "multicolor"
            // path (two-nearest selection over the whole palette).
            PrintMan::ColorField color3;
            color3.palette     = {FlushPredict::RGBColor(255, 0, 0), FlushPredict::RGBColor(0, 255, 0), FlushPredict::RGBColor(0, 0, 255)};
            color3.dither      = true;
            color3.band_width  = 0.5;
            color3.dither_cell = 0.5;
            color3.eval        = [](const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &) {
                return PrintMan::V3{{0.5, 0.5, 0.0}};   // red+green; blue is the odd one out
            };
            std::vector<std::vector<ExPolygons>> seg;
            PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, {}, color3, &seg);
            const double a0 = total(seg, 0), a1 = total(seg, 1), a2 = total(seg, 2);
            THEN("the two relevant filaments dither and the irrelevant one is unused") {
                REQUIRE(a0 > 0.0);          // red
                REQUIRE(a1 > 0.0);          // green
                REQUIRE(a2 == 0.0);         // blue never chosen -- not among the two nearest
                REQUIRE(std::abs(a0 - a1) < 0.5 * (a0 + a1));
            }
        }
    }
}

// A continuous colour gradient must span the WHOLE object bottom-to-top, dithered up the side -- the
// feature Karl asked to see. The colour is object-normalized in Z (t = 0 at the bottom layer, 1 at the
// top) exactly as PrintObjectSlice's colorObjectSpace wrapper does, so the gradient scales with the
// object at any height and never saturates (the old "all cyan" bug). Cout = magenta at the base, cyan at
// the top; dithered, so at height t a fraction ~t of the wall goes to cyan. Assert the cyan share climbs
// monotonically from ~0 at the bottom to ~1 at the top -- the gradient spans the object. C++ lambda, no OSL.
SCENARIO_METHOD(UsdResourcesFixture, "A dithered colour gradient spans the whole object bottom to top", "[usd][subdiv][PrintMan][color][gradient]")
{
    GIVEN("a subdivision cube shaded magenta(bottom)->cyan(top) over a magenta/cyan 2-filament palette") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, true));
        const ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());

        const Transform3d m = vol->get_matrix();
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const Vec3f &v : vol->mesh().its.vertices) {
            const float z = float((m * v.cast<double>()).z());
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
            zs.push_back(z);
        REQUIRE(zs.size() > 10);

        MeshSlicingParamsEx params;
        params.trafo      = m;
        params.subdiv_tol = 0.05;

        // Object-normalized gradient: t in [0,1] across the object's Z, magenta(1,0,1)->cyan(0,1,1) linear.
        const double z0 = zs.front(), zspan = double(zs.back()) - z0;
        PrintMan::ColorField color;
        color.palette     = {FlushPredict::RGBColor(255, 0, 255), FlushPredict::RGBColor(0, 255, 255)};
        color.dither      = true;
        color.band_width  = 0.5;   // match the dither cell so ribbons barely overlap (no lower-id bias)
        color.dither_cell = 0.5;
        color.eval        = [z0, zspan](const PrintMan::V3 &p, const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &) {
            double t = zspan > 1e-9 ? (p[2] - z0) / zspan : 0.0;
            t = std::clamp(t, 0.0, 1.0);
            return PrintMan::V3{{1.0 - t, t, 1.0}};   // channel 0 = magenta, channel 1 = cyan
        };

        std::vector<std::vector<ExPolygons>> seg;
        PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, {}, color, &seg);
        REQUIRE(seg.size() == zs.size());

        auto area = [](const ExPolygons &e) { double a = 0.0; for (const ExPolygon &p : e) a += p.area(); return a; };

        THEN("the cyan share climbs from ~0 at the bottom to ~1 at the top -- a gradient over the whole object") {
            // Bin the layers into fifths of the object's height; per bin, cyan's share of the coloured area.
            const int    NB = 5;
            double       mag[NB] = {0}, cyan[NB] = {0};
            for (size_t L = 0; L < zs.size(); ++ L) {
                REQUIRE(seg[L].size() == 2);
                double t = zspan > 1e-9 ? (double(zs[L]) - z0) / zspan : 0.0;
                int    b = std::min(NB - 1, std::max(0, int(t * NB)));
                mag[b]  += area(seg[L][0]);
                cyan[b] += area(seg[L][1]);
            }
            std::vector<double> frac(NB, 0.0);
            for (int b = 0; b < NB; ++ b) {
                const double tot = mag[b] + cyan[b];
                frac[b] = tot > 0.0 ? cyan[b] / tot : -1.0;
                WARN("height bin " << b << " cyan_share=" << frac[b] << " (magenta=" << mag[b] << " cyan=" << cyan[b] << ")");
                REQUIRE(tot > 0.0);   // every height band of the object is coloured -- the gradient spans it
            }
            REQUIRE(frac[0]      < 0.25);          // bottom is almost all magenta
            REQUIRE(frac[NB - 1] > 0.75);          // top is almost all cyan
            REQUIRE(frac[NB - 1] - frac[0] > 0.6); // and it genuinely traverses the palette up the side
            for (int b = 1; b < NB; ++ b)
                REQUIRE(frac[b] >= frac[b - 1] - 0.1);   // monotonic up to dither noise -- no reversal
        }
    }
}

// Full multi-filament dithering: a surface colour that sweeps through the WHOLE loaded palette up the
// object must dither across ALL of it, each filament landing at the height where the colour matches it.
// The field is a piecewise-linear sweep (in linear light) through the six palette colours in order, so at
// height k/5 the colour IS filament k and between them the two neighbours dither. Assert every filament is
// used and their area-weighted mean heights are strictly ordered red<...<magenta -- the colour climbs the
// whole palette up the side. C++ lambda, no OSL, always built.
SCENARIO_METHOD(UsdResourcesFixture, "The engine dithers a surface colour across all loaded filaments", "[usd][subdiv][PrintMan][color][dither][allfilament]")
{
    GIVEN("a subdivision cube swept through a 6-filament palette (red->magenta) up its height") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, true));
        const ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());

        const Transform3d m = vol->get_matrix();
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const Vec3f &v : vol->mesh().its.vertices) {
            const float z = float((m * v.cast<double>()).z());
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
            zs.push_back(z);
        REQUIRE(zs.size() > 10);

        MeshSlicingParamsEx params;
        params.trafo      = m;
        params.subdiv_tol = 0.05;

        const std::vector<FlushPredict::RGBColor> pal = {
            {255, 0, 0}, {255, 255, 0}, {0, 255, 0}, {0, 255, 255}, {0, 0, 255}, {255, 0, 255}};
        const int NP = int(pal.size());
        std::vector<PrintMan::V3> lin;
        for (const auto &c : pal) lin.push_back(PrintMan::linear_from_srgb(c));

        const double z0 = zs.front(), zspan = double(zs.back()) - z0;
        PrintMan::ColorField color;
        color.palette     = pal;
        color.dither      = true;
        color.band_width  = 0.5;
        color.dither_cell = 0.5;
        color.eval        = [z0, zspan, lin, NP](const PrintMan::V3 &p, const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &) {
            double t = zspan > 1e-9 ? (p[2] - z0) / zspan : 0.0;
            t = std::clamp(t, 0.0, 1.0);
            const double u = t * (NP - 1);
            const int    i = std::min(NP - 2, int(u));
            const double f = u - i;
            return PrintMan::V3{{(1 - f) * lin[i][0] + f * lin[i + 1][0],
                                 (1 - f) * lin[i][1] + f * lin[i + 1][1],
                                 (1 - f) * lin[i][2] + f * lin[i + 1][2]}};
        };

        std::vector<std::vector<ExPolygons>> seg;
        PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, {}, color, &seg);
        REQUIRE(seg.size() == zs.size());

        auto area = [](const ExPolygons &e) { double a = 0.0; for (const ExPolygon &p : e) a += p.area(); return a; };

        THEN("every filament is used, and each one's mean height climbs in palette order") {
            std::vector<double> wsum(NP, 0.0), asum(NP, 0.0);
            for (size_t L = 0; L < zs.size(); ++ L) {
                REQUIRE(int(seg[L].size()) == NP);
                const double t = zspan > 1e-9 ? (double(zs[L]) - z0) / zspan : 0.0;
                for (int k = 0; k < NP; ++ k) {
                    const double a = area(seg[L][k]);
                    asum[k] += a;
                    wsum[k] += a * t;
                }
            }
            std::vector<double> meanz(NP, 0.0);
            for (int k = 0; k < NP; ++ k) {
                WARN("filament " << k << " total_area=" << asum[k] << " mean_height=" << (asum[k] > 0 ? wsum[k] / asum[k] : -1.0));
                REQUIRE(asum[k] > 0.0);                 // EVERY loaded filament is used somewhere
                meanz[k] = wsum[k] / asum[k];
            }
            for (int k = 1; k < NP; ++ k)
                REQUIRE(meanz[k] > meanz[k - 1] + 0.05);   // strictly climbing -> the sweep spans the palette in order
        }
    }
}

// What Karl SEES on the printed side is the outer wall's colour, not the interior (that stays the base
// filament). The colour is deposited as a band_width-wide ribbon along each face's slice segment; if the
// ribbon is thinner than a perimeter it fails to claim the outer wall and the object reads as one filament
// even though the segmentation has every channel. Measure the fraction of a one-perimeter-wide outer ring
// that the coloured ribbons actually cover, swept over band_width. Proves a too-thin ribbon (my
// band_width = dither_cell regression) under-covers, and a >= 1-perimeter ribbon colours the wall.
SCENARIO_METHOD(UsdResourcesFixture, "A dithered surface colour claims the object's outer wall", "[usd][subdiv][PrintMan][color][coverage]")
{
    GIVEN("a gradient-shaded cube sliced with colour ribbons of different widths") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, true));
        const ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());

        const Transform3d m = vol->get_matrix();
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const Vec3f &v : vol->mesh().its.vertices) {
            const float z = float((m * v.cast<double>()).z());
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f) zs.push_back(z);
        REQUIRE(zs.size() > 10);
        MeshSlicingParamsEx params; params.trafo = m; params.subdiv_tol = 0.05;
        const double z0 = zs.front(), zspan = double(zs.back()) - z0;

        auto area = [](const ExPolygons &e) { double a = 0.0; for (const ExPolygon &p : e) a += p.area(); return a; };
        // Fraction of a ~one-perimeter outer ring covered by the union of coloured ribbons, at ribbon width bw.
        auto outer_wall_coverage = [&](double bw) {
            PrintMan::ColorField color;
            color.palette     = {FlushPredict::RGBColor(255, 0, 255), FlushPredict::RGBColor(0, 255, 255)};
            color.dither      = true;
            color.dither_cell = 0.5;
            color.band_width  = bw;
            color.eval        = [z0, zspan](const PrintMan::V3 &p, const PrintMan::V3 &, const PrintMan::V3 &, const PrintMan::V3 &) {
                double t = zspan > 1e-9 ? (p[2] - z0) / zspan : 0.0; t = std::clamp(t, 0.0, 1.0);
                return PrintMan::V3{{1.0 - t, t, 1.0}};
            };
            std::vector<std::vector<ExPolygons>> seg;
            const std::vector<ExPolygons> layers = PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, {}, color, &seg);
            double ring_area = 0.0, colored_ring = 0.0;
            for (size_t L = 0; L < layers.size(); ++ L) {
                if (layers[L].empty()) continue;
                const ExPolygons inner = offset_ex(layers[L], - float(scale_(0.45)));   // erode by ~one perimeter
                const ExPolygons ring  = diff_ex(layers[L], inner);
                ExPolygons all;
                for (const ExPolygons &ch : seg[L]) all.insert(all.end(), ch.begin(), ch.end());
                const ExPolygons colored = union_ex(all);
                ring_area    += area(ring);
                colored_ring += area(intersection_ex(colored, ring));
            }
            return ring_area > 0.0 ? colored_ring / ring_area : 0.0;
        };

        const double c05 = outer_wall_coverage(0.5);   // the band_width = dither_cell regression
        const double c10 = outer_wall_coverage(1.0);   // the ColorField default (what the grid demo used)
        const double c15 = outer_wall_coverage(1.5);
        WARN("outer-wall colour coverage: band=0.5 -> " << c05 << ", band=1.0 -> " << c10 << ", band=1.5 -> " << c15);
        THEN("a ribbon at least a perimeter wide colours most of the outer wall; the too-thin one does not") {
            REQUIRE(c10 > 0.85);           // the default 1.0 mm ribbon covers the visible side
            REQUIRE(c10 > c05 + 0.2);      // materially better than the thin ribbon that reads as one filament
        }
    }
}

SCENARIO_METHOD(UsdResourcesFixture, "A displacement shader moves the sliced surface", "[usd][subdiv][PrintMan][displace]")
{
    GIVEN("a subdivision cube sliced with and without a raised Gridwork displacement") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, true));
        const ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());
        REQUIRE(vol->printman_scene->cages.size() == 1);

        const Transform3d m = vol->get_matrix();
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const Vec3f &v : vol->mesh().its.vertices) {
            const float z = float((m * v.cast<double>()).z());
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
            zs.push_back(z);
        REQUIRE(zs.size() > 10);

        MeshSlicingParamsEx params;
        params.trafo      = m;
        params.subdiv_tol = 0.05;

        // A raised gridwork displacement, evaluated in world space (mm).
        PrintMan::Device            dev;
        PrintMan::Gridwork          gw;
        PrintMan::DisplacementField disp;
        disp.max_magnitude = gw.effective_depth(dev);
        disp.eval = [gw, dev](const PrintMan::V3 &p, const PrintMan::V3 &n) { return gw(p, n, dev); };

        const std::vector<ExPolygons> plain    = PrintMan::slice_scene(*vol->printman_scene, params, zs);
        const std::vector<ExPolygons> textured = PrintMan::slice_scene(*vol->printman_scene, params, zs,
                                                                       [](){}, {}, disp);

        REQUIRE(plain.size()    == zs.size());
        REQUIRE(textured.size() == zs.size());

        auto total_area = [](const std::vector<ExPolygons> &layers) {
            double a = 0.0;
            for (const ExPolygons &layer : layers)
                for (const ExPolygon &ep : layer)
                    a += ep.area();
            return a;
        };
        const double a_plain    = total_area(plain);
        const double a_textured = total_area(textured);

        THEN("both slice to valid, non-empty contours") {
            REQUIRE(a_plain    > 0.0);
            REQUIRE(a_textured > 0.0);
        }
        THEN("displacement changes the geometry, and a raised gridwork grows it outward") {
            REQUIRE(std::abs(a_textured - a_plain) > 0.02 * a_plain);   // measurably different
            REQUIRE(a_textured > a_plain);                             // raised => pushed out
        }
    }
}

SCENARIO_METHOD(UsdResourcesFixture, "Multi-band displacement stays seam-continuous", "[usd][subdiv][PrintMan][displace][seam]")
{
    GIVEN("a tall prism (many bands) sliced with a low-amplitude noise (max_disp < layer height)") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("tall_prism_catmull.usda").c_str(), &model, message, nullptr, true));
        const ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());

        const Transform3d m = vol->get_matrix();
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const Vec3f &v : vol->mesh().its.vertices) {
            const float z = float((m * v.cast<double>()).z());
            zmin = std::min(zmin, z);
            zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
            zs.push_back(z);
        REQUIRE(zs.size() > 100);   // many layers => many bands

        MeshSlicingParamsEx params;
        params.trafo      = m;
        params.subdiv_tol = 0.05;

        // Low amplitude: max |disp| = 0.3 * 0.42 = 0.126 mm < 0.2 mm layer -> the seam-cracking regime.
        PrintMan::Device dev;
        PrintMan::Noise  noise;
        noise.amplitude_beads = 0.3;
        PrintMan::DisplacementField disp;
        disp.max_magnitude = dev.beads(noise.amplitude_beads);
        disp.eval = [noise, dev](const PrintMan::V3 &p, const PrintMan::V3 &n) { return noise(p, n, dev); };

        size_t bands = 0;   // the cage's band count (progress total) -- prove this is multi-band
        const std::vector<ExPolygons> out = PrintMan::slice_scene(*vol->printman_scene, params, zs,
            [](){}, [&](size_t, size_t total){ bands = total; }, disp);
        REQUIRE(out.size() == zs.size());
        REQUIRE(bands > 3);   // precondition: the prism really is sliced in several bands

        // Adjacent layers are 0.2 mm apart; on a smooth low-amplitude field the cross-section varies
        // smoothly. A per-band-normal seam is a lateral step -> a spike in the adjacent-layer area jump.
        std::vector<double> area(zs.size(), 0.0);
        for (size_t i = 0; i < zs.size(); ++i)
            for (const ExPolygon &ep : out[i]) area[i] += ep.area();
        // Isolate the MIDDLE (vertical side faces, where band seams live) from the top/bottom
        // horizontal faces (whose Z-displacement/clip is a separate, documented artifact).
        auto max_jump_in = [&](double zlo, double zhi, size_t &at) {
            double mj = 0.0;
            for (size_t i = 1; i < zs.size(); ++i) {
                if (zs[i] < zlo || zs[i] > zhi || area[i] <= 0.0 || area[i - 1] <= 0.0) continue;
                const double j = std::abs(area[i] - area[i - 1]) / area[i - 1];
                if (j > mj) { mj = j; at = i; }
            }
            return mj;
        };
        size_t at_mid = 0;
        const double mid = max_jump_in(8.0, 32.0, at_mid);
        THEN("the band seams on the vertical side faces stay continuous") {
            // The mid-prism side faces cross several band boundaries here. A per-band-normal seam
            // would spike the adjacent-layer area; empirically it stays smooth (~1%) for this coarse
            // cage even with max_disp (0.126mm) < the 0.2mm layer. So the band-invariant-normal
            // hardening the review called for is a corner case -- fine tessellation AND low amplitude
            // together -- not a common-case break. (The top/bottom HORIZONTAL faces DO step badly:
            // +/-Z relief is clipped by Orca's fixed layer set and unseats the part -- a separate,
            // documented known limit, deliberately not asserted here.)
            INFO("mid-region (8..32mm) max adjacent-area jump = " << mid);
            CHECK(mid < 0.05);
        }
    }
}

SCENARIO_METHOD(UsdResourcesFixture, "Reading a USD file", "[usd]")
{
    GIVEN("a stage declaring metersPerUnit and a Y up-axis") {
        THEN("both are applied") {
            Model       model;
            std::string message;
            REQUIRE(load_usd(usd_path("cube_metres.usda").c_str(), &model, message));
            REQUIRE(is_approx(model.objects.front()->volumes.front()->mesh().size(),
                              Vec3d(10000, 10000, 10000)));

            // Y-up import: rotated under the plate, same extents.
            TriangleMesh mesh;
            REQUIRE(load_usd(usd_path("cube_yup.usda").c_str(), &mesh, message));
            REQUIRE(mesh.bounding_box().min.z() == Catch::Approx(0.0).margin(1e-4));
            REQUIRE(mesh.bounding_box().max.z() == Catch::Approx(10.0).margin(1e-4));
        }
    }

    // Drops to .usda alone and nothing covers the plugInfo staging any more.
    GIVEN("the same cube as text, as a binary crate and as a usdz package") {
        THEN("all three import identically") {
            for (const char *f : {"cube_none.usda", "cube_none.usdc", "cube_none.usdz"}) {
                Model       model;
                std::string message;
                REQUIRE(load_usd(usd_path(f).c_str(), &model, message));
                REQUIRE(model.objects.front()->volumes.front()->mesh().facets_count() == 12);
            }
        }
    }

    GIVEN("a catmullClark subdivision cage, imported eagerly") {
        THEN("it is refined to its subdivision surface, not imported flat") {
            Model       model;
            std::string message;
            // amplify=false refines at import (reference path); amplify=true defers to slice time (below).
            REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, false));
            const TriangleMesh &mesh = model.objects.front()->volumes.front()->mesh();
            // 6 faces -> Catmull-Clark level 2 -> 96 quads -> 192 triangles (12 unsubdivided).
            REQUIRE(mesh.facets_count() == 192);
            // The limit surface contracts inward, so it sits inside the 10 mm control cube.
            REQUIRE(mesh.bounding_box().min.x() > -5.0);
            REQUIRE(mesh.bounding_box().max.x() <  5.0);
        }
    }
}

// Every world-space vertex of a model, sorted, so two imports are comparable regardless of
// centering. PrintMan geometry is prototype-through-placement; a flat volume's is its mesh.
static std::vector<Vec3d> world_vertices(const Model &model)
{
    std::vector<Vec3d> pts;
    for (const ModelVolume *v : model.objects.front()->volumes) {
        const Transform3d m = v->get_matrix();
        if (v->printman_scene) {
            for (const PrintMan::Placement &pl : v->printman_scene->placements) {
                const indexed_triangle_set &proto = v->printman_scene->prototypes[pl.prototype];
                for (const Vec3f &p : proto.vertices)
                    pts.push_back(m * pl.xform * p.cast<double>());
            }
        } else {
            for (const Vec3f &p : v->mesh().its.vertices)
                pts.push_back(m * p.cast<double>());
        }
    }
    std::sort(pts.begin(), pts.end(), [](const Vec3d &a, const Vec3d &b) {
        if (a.x() != b.x()) return a.x() < b.x();
        if (a.y() != b.y()) return a.y() < b.y();
        return a.z() < b.z();
    });
    return pts;
}

// ORIENTED world triangles the two models share (connectivity + winding, which the sorted vertex
// SET in world_vertices() can't see). Vertices are welded by proximity, so the ~1e-6 mm path gap
// can't split a match. Exact only when the prototype has no reader-samplable symmetry.
struct TriDiff { size_t matched, a_only, b_only; };
static TriDiff world_triangle_diff(const Model &a, const Model &b)
{
    std::map<std::array<long, 3>, std::vector<std::pair<Vec3d, int>>> grid;
    const double CELL = 0.05;    // 50 um cells
    const double TOL2 = 1e-6;    // (1e-3 mm)^2 weld radius: >> path gap, << feature spacing
    int next = 0;
    auto canon = [&](const Vec3d &p) {
        std::array<long, 3> c{ long(std::floor(p.x() / CELL)),
                               long(std::floor(p.y() / CELL)),
                               long(std::floor(p.z() / CELL)) };
        for (long dx = -1; dx <= 1; ++ dx)
            for (long dy = -1; dy <= 1; ++ dy)
                for (long dz = -1; dz <= 1; ++ dz) {
                    auto it = grid.find({ c[0] + dx, c[1] + dy, c[2] + dz });
                    if (it == grid.end()) continue;
                    for (const auto &pr : it->second)
                        if ((pr.first - p).squaredNorm() < TOL2) return pr.second;
                }
        int id = next ++;
        grid[c].push_back({ p, id });
        return id;
    };
    auto keys = [&](const Model &m) {
        std::vector<std::array<int, 3>> ks;
        auto emit = [&](const indexed_triangle_set &its, const Transform3d &X) {
            for (const auto &t : its.indices) {
                std::array<int, 3> id{ canon(X * its.vertices[t[0]].template cast<double>()),
                                       canon(X * its.vertices[t[1]].template cast<double>()),
                                       canon(X * its.vertices[t[2]].template cast<double>()) };
                int s = 0;                                 // rotate smallest-first, keep cycle
                for (int i = 1; i < 3; ++ i) if (id[i] < id[s]) s = i;
                ks.push_back({ id[s], id[(s + 1) % 3], id[(s + 2) % 3] });
            }
        };
        for (const ModelVolume *v : m.objects.front()->volumes) {
            const Transform3d M = v->get_matrix();
            if (v->printman_scene)
                for (const PrintMan::Placement &pl : v->printman_scene->placements)
                    emit(v->printman_scene->prototypes[pl.prototype], M * pl.xform);
            else
                emit(v->mesh().its, M);
        }
        std::sort(ks.begin(), ks.end());
        return ks;
    };
    const auto ka = keys(a), kb = keys(b);   // kb welds onto ka's canonical ids
    TriDiff d{ 0, 0, 0 };
    size_t i = 0, j = 0;
    while (i < ka.size() && j < kb.size()) {
        if      (ka[i] == kb[j]) { ++ d.matched; ++ i; ++ j; }
        else if (ka[i] <  kb[j]) { ++ d.a_only;  ++ i; }
        else                     { ++ d.b_only;  ++ j; }
    }
    d.a_only += ka.size() - i;
    d.b_only += kb.size() - j;
    return d;
}

SCENARIO_METHOD(UsdResourcesFixture, "USD instancing imports as a PrintMan scene", "[usd][PrintMan]")
{
    GIVEN("an instanced stage with a rotated and a scaled instance") {
        Model       flat, scene;
        std::string message;
        // Same stage two ways: amplify=false (trusted native-transform flatten) vs amplify=true (scene path).
        REQUIRE(load_usd(usd_path("instanced_transformed.usda").c_str(), &flat,  message, nullptr, false));
        REQUIRE(load_usd(usd_path("instanced_transformed.usda").c_str(), &scene, message, nullptr, true));

        THEN("the scene has one deduped prototype and three placements") {
            REQUIRE(scene.objects.front()->volumes.size() == 1);
            const ModelVolume *vol = scene.objects.front()->volumes.front();
            REQUIRE(vol->printman_scene.has_value());
            REQUIRE(vol->printman_scene->prototypes.size() == 1);
            REQUIRE(vol->printman_scene->placements.size() == 3);
            REQUIRE(flat.objects.front()->volumes.size() == 3);   // flatten = one per instance
            // A 12-facet box per instance (not the real geometry); the world-vertex check below is the gate.
            REQUIRE(vol->mesh().facets_count() == 12 * vol->printman_scene->placements.size());
        }

        THEN("scene and flatten place identical world geometry (so the transpose is right)") {
            // A wrong row->column transpose is a no-op on translations but diverges on the rotated/scaled instances.
            const std::vector<Vec3d> sv = world_vertices(scene);
            const std::vector<Vec3d> fv = world_vertices(flat);
            REQUIRE(sv.size() == fv.size());
            double max_diff = 0.0;
            for (size_t i = 0; i < sv.size(); ++ i)
                max_diff = std::max(max_diff, (sv[i] - fv[i]).cwiseAbs().maxCoeff());
            REQUIRE(max_diff < 1e-3);
        }
    }
}

SCENARIO_METHOD(UsdResourcesFixture, "An instanced USD slices amplified through Orca's pipeline", "[usd][PrintMan]")
{
    GIVEN("the eight-cube instanced stage imported with amplification") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(usd_path("instanced_cubes.usda").c_str(), &model, message, nullptr, true));
        ModelVolume *vol = model.objects.front()->volumes.front();
        REQUIRE(vol->printman_scene.has_value());
        REQUIRE(vol->printman_scene->placements.size() == 8);

        // Minimal print setup, mirroring tests/fff_print/test_helpers.cpp::init_print.
        DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
        ModelObject *mo = model.objects.front();
        mo->add_instance();
        arrange_objects(model, arrangement::InfiniteBed{}, arrangement::ArrangeParams{ scaled(min_object_distance(config)) });
        mo->ensure_on_bed();
        Print print;
        print.auto_assign_extruders(mo);
        print.apply(model, config);
        print.validate();

        THEN("Orca's own pipeline slices it into eight amplified islands") {
            std::string err;
            try { print.process(); }
            catch (const SlicingErrors &e) { for (const auto &se : e.errors_) err += std::string("[") + se.what() + "]"; }
            catch (const SlicingError  &e) { err = e.what(); }
            catch (const std::exception &e) { err = e.what(); }
            INFO("process errors: " << err);
            REQUIRE(err.empty());
            const auto layers = print.objects().front()->layers();
            REQUIRE(layers.size() > 0);
            REQUIRE(layers[layers.size() / 2]->lslices.size() == 8);
        }
    }
}

static double total_area(const ExPolygons &e)
{
    double a = 0.0;
    for (const ExPolygon &p : e) a += p.area();
    return a;
}

// Symmetric-difference area of two per-layer regions; zero iff identical.
static double xor_area(const ExPolygons &a, const ExPolygons &b)
{
    return total_area(diff_ex(a, b)) + total_area(diff_ex(b, a));
}

// Take a freshly-loaded instanced model through Orca's pipeline, identically regardless of how
// it was loaded, placed deterministically so a positioning divergence can't masquerade as
// slicing agreement.
static void process_instanced(Model &model, Print &print)
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    ModelObject *mo = model.objects.front();
    mo->add_instance();
    // Deterministic identical placement for both (not arrange_objects: it could ROTATE-to-pack the
    // two differently -- they have different 2D hulls -- making the contour comparison meaningless).
    mo->ensure_on_bed();
    const Vec3d c = mo->bounding_box_exact().center();
    ModelInstance *inst = mo->instances.front();
    inst->set_offset(inst->get_offset() + Vec3d(150.0 - c.x(), 150.0 - c.y(), 0.0));
    mo->ensure_on_bed();
    print.auto_assign_extruders(mo);
    print.apply(model, config);
    try {
        print.process();
    } catch (const SlicingErrors &e) {
        std::string err;
        for (const auto &se : e.errors_) err += std::string("[") + se.what() + "]";
        std::cerr << "[equiv] process SlicingErrors: " << err << std::endl;
        throw;
    } catch (const SlicingError &e) {
        std::cerr << "[equiv] process SlicingError: " << e.what() << std::endl;
        throw;
    }
}

// The amplified path and the naive flatten must slice the same file to the same part, compared
// layer-for-layer by symmetric-difference area. torus_vertical is a hole-bearing, tilted, welded
// torus; PRINTMAN_USD=<file> overrides the fixtures.
SCENARIO_METHOD(UsdResourcesFixture, "Amplified slicing matches naive flattening layer-for-layer",
                "[usd][PrintMan]")
{
    std::vector<std::string> files;
    if (const char *env = std::getenv("PRINTMAN_USD"))
        files.emplace_back(env);
    else {
        files.push_back(usd_path("instanced_transformed.usda"));
        files.push_back(usd_path("instanced_cubes.usda"));
        files.push_back(usd_path("torus_vertical.usda"));
    }

    for (const std::string &f : files) {
        DYNAMIC_SECTION("file: " << f) {
            Model       ours, naive;
            std::string message;
            REQUIRE(load_usd(f.c_str(), &ours,  message, nullptr, true));   // amplified scene
            REQUIRE(load_usd(f.c_str(), &naive, message, nullptr, false));  // flattened copies

            // Same geometry => same bounding box, else arrange places them differently.
            const BoundingBoxf3 bb_o = ours.objects.front()->bounding_box_exact();
            const BoundingBoxf3 bb_n = naive.objects.front()->bounding_box_exact();
            REQUIRE((bb_o.min - bb_n.min).cwiseAbs().maxCoeff() < 1e-3);
            REQUIRE((bb_o.max - bb_n.max).cwiseAbs().maxCoeff() < 1e-3);

            // Same world geometry into the slicer? A divergence here is the reader/transform, not slicing.
            {
                const std::vector<Vec3d> ov = world_vertices(ours), nv = world_vertices(naive);
                double wd = -1.0;
                if (ov.size() == nv.size()) {
                    wd = 0.0;
                    for (size_t i = 0; i < ov.size(); ++ i)
                        wd = std::max(wd, (ov[i] - nv[i]).cwiseAbs().maxCoeff());
                }
                std::cerr << "[equiv] world-verts ours=" << ov.size() << " naive=" << nv.size()
                          << " maxdiff=" << wd << std::endl;
            }

            // ...and the same triangles (connectivity + winding); asserted on fixtures, logged for a PRINTMAN_USD override.
            {
                const TriDiff td = world_triangle_diff(ours, naive);
                std::cerr << "[equiv] world-tris matched=" << td.matched
                          << " ours_only=" << td.a_only << " naive_only=" << td.b_only << std::endl;
                if (! std::getenv("PRINTMAN_USD")) {
                    REQUIRE(td.a_only == 0);
                    REQUIRE(td.b_only == 0);
                }
            }

            Print p_ours, p_naive;
            process_instanced(ours,  p_ours);
            process_instanced(naive, p_naive);

            const auto La = p_ours.objects().front()->layers();
            const auto Lb = p_naive.objects().front()->layers();
            REQUIRE(La.size() == Lb.size());
            REQUIRE(La.size() > 0);

            double max_rel  = 0.0;
            size_t worst    = 0;
            bool   isl_diff = false;
            double sum_area = 0.0;
            for (size_t i = 0; i < La.size(); ++ i) {
                const ExPolygons &a = La[i]->lslices;
                const ExPolygons &b = Lb[i]->lslices;
                if (a.size() != b.size()) isl_diff = true;
                const double denom = std::max(total_area(a), total_area(b));
                sum_area += total_area(a);
                if (denom <= 0.0) continue;
                const double rel = xor_area(a, b) / denom;
                if (rel > max_rel) { max_rel = rel; worst = i; }
            }
            const ExPolygons &wa = La[worst]->lslices;
            const ExPolygons &wb = Lb[worst]->lslices;
            std::cerr << "[equiv] " << f << " layers=" << La.size()
                      << " islands(mid)=" << La[La.size() / 2]->lslices.size()
                      << " max_rel_xor=" << max_rel << " @layer " << worst
                      << " island_count_mismatch=" << (isl_diff ? "YES" : "no")
                      << " | worstL ours[n=" << wa.size() << " area=" << total_area(wa)
                      << "] naive[n=" << wb.size() << " area=" << total_area(wb) << "]"
                      << std::endl;

            REQUIRE(sum_area > 0.0);            // there is geometry to compare
            REQUIRE_FALSE(isl_diff);            // same island count on every layer
            REQUIRE(max_rel < 1e-3);            // contours agree to <0.1% of layer area
        }
    }
}

// catmark_cube_creases0 (OpenSubdiv regression/shapes/): a 45-deg cube with a sharpness-2.0 crease.
// CREASES0_GOLDEN_L2 is it refined to CC level 2 by OpenSubdiv's own Far library (via tools/osd), so
// this pins our evaluator to Pixar's reference, not our arithmetic; the tolerance below is above the
// ~6.5e-8 float32-vs-double residual and below the feature spacing.
static const double CREASES0_GOLDEN_L2[98][3] = {
    {     0.02946279,    -0.74966437,     0.55092597 },
    {     0.97227210,     0.00000000,     1.00000000 },
    {    -0.72020155,    -0.00000000,     0.50925928 },
    {     0.02946279,     0.74966437,     0.55092597 },
    {    -0.72020155,     0.00000000,    -0.50925928 },
    {    -0.00000000,     0.72020155,    -0.50925928 },
    {    -0.00000000,    -0.72020155,    -0.50925928 },
    {     0.72020155,    -0.00000000,    -0.50925928 },
    {     0.03744231,     0.00000000,     0.93229163 },
    {    -0.62117386,     0.62117386,     0.00000000 },
    {     0.00000000,     0.00000000,    -0.87847221 },
    {     0.64204335,    -0.63774669,     0.03038194 },
    {     0.64204335,     0.63774669,     0.03038195 },
    {    -0.62117386,    -0.62117386,     0.00000000 },
    {     0.66291285,    -0.62853956,     0.94444442 },
    {     0.66291285,     0.62853956,     0.94444442 },
    {    -0.45206970,     0.45759398,     0.64713544 },
    {    -0.45206970,    -0.45759398,     0.64713544 },
    {     0.00276214,     0.91242588,     0.00390625 },
    {    -0.45483184,     0.45483184,    -0.64322919 },
    {    -0.90966368,     0.00000000,     0.00000000 },
    {     0.45483184,     0.45483184,    -0.64322919 },
    {     0.45483184,    -0.45483184,    -0.64322919 },
    {    -0.45483184,    -0.45483184,    -0.64322919 },
    {     0.94096792,    -0.00000000,     0.04947916 },
    {     0.00276214,    -0.91242588,     0.00390625 },
    {     0.04419419,    -0.50577790,     0.82638890 },
    {     0.61871862,     0.00000000,     1.00000000 },
    {     0.04419419,     0.50577790,     0.82638890 },
    {    -0.46158373,     0.00000000,     0.76388890 },
    {    -0.77094305,     0.30935931,     0.32638890 },
    {    -0.30935931,     0.77094305,     0.32638890 },
    {    -0.30935931,     0.77094305,    -0.32638890 },
    {    -0.77094305,     0.30935931,    -0.32638890 },
    {    -0.46158373,     0.00000000,    -0.76388890 },
    {     0.00000000,     0.46158373,    -0.76388890 },
    {     0.46158373,     0.00000000,    -0.76388890 },
    {     0.00000000,    -0.46158373,    -0.76388890 },
    {     0.30935931,    -0.77094305,    -0.32638890 },
    {     0.77094305,    -0.30935931,    -0.32638890 },
    {     0.88388371,    -0.35355350,     0.50000000 },
    {     0.35355350,    -0.81513727,     0.38888890 },
    {     0.88388377,     0.35355350,     0.50000000 },
    {     0.77094305,     0.30935931,    -0.32638890 },
    {     0.30935931,     0.77094305,    -0.32638890 },
    {     0.35355350,     0.81513727,     0.38888890 },
    {    -0.30935931,    -0.77094305,    -0.32638890 },
    {    -0.30935931,    -0.77094305,     0.32638890 },
    {    -0.77094305,    -0.30935931,     0.32638890 },
    {    -0.77094305,    -0.30935931,    -0.32638890 },
    {     0.34250495,    -0.30322123,     0.95659721 },
    {     0.34250498,     0.30322123,     0.95659721 },
    {    -0.23692995,     0.25902703,     0.83506942 },
    {    -0.23692995,    -0.25902703,     0.83506942 },
    {    -0.57943493,     0.57943487,     0.35069442 },
    {    -0.33145642,     0.82741344,     0.00000000 },
    {    -0.57943487,     0.57943493,    -0.35069442 },
    {    -0.82741344,     0.33145642,     0.00000000 },
    {    -0.24797849,     0.24797849,    -0.81944442 },
    {     0.24797849,     0.24797849,    -0.81944442 },
    {     0.24797849,    -0.24797849,    -0.81944442 },
    {    -0.24797849,    -0.24797849,    -0.81944442 },
    {     0.57943487,    -0.57943493,    -0.35069442 },
    {     0.85564858,    -0.34250498,     0.04340278 },
    {     0.66291279,    -0.64572620,     0.47222221 },
    {     0.34250495,    -0.83846200,     0.01562500 },
    {     0.85564852,     0.34250495,     0.04340278 },
    {     0.57943493,     0.57943487,    -0.35069442 },
    {     0.34250498,     0.83846200,     0.01562500 },
    {     0.66291285,     0.64572620,     0.47222221 },
    {    -0.33145642,    -0.82741344,     0.00000000 },
    {    -0.57943487,    -0.57943493,     0.35069442 },
    {    -0.82741344,    -0.33145642,     0.00000000 },
    {    -0.57943493,    -0.57943487,    -0.35069442 },
    {     0.35355350,    -0.74639070,     0.77777779 },
    {     0.88388371,    -0.35355350,     1.00000000 },
    {     0.88388371,     0.35355350,     1.00000000 },
    {     0.35355350,     0.74639070,     0.77777779 },
    {    -0.19887385,     0.64818144,     0.61458331 },
    {    -0.63713288,     0.20992239,     0.59895831 },
    {    -0.63713288,    -0.20992239,     0.59895831 },
    {    -0.19887385,    -0.64818144,     0.61458331 },
    {     0.01104855,     0.85810387,     0.31770834 },
    {     0.00000000,     0.84705532,    -0.30208334 },
    {    -0.20992239,     0.63713288,    -0.59895831 },
    {    -0.63713288,     0.20992239,    -0.59895831 },
    {    -0.84705532,     0.00000000,    -0.30208334 },
    {    -0.84705532,     0.00000000,     0.30208334 },
    {     0.20992239,     0.63713288,    -0.59895831 },
    {     0.63713288,     0.20992239,    -0.59895831 },
    {     0.63713288,    -0.20992239,    -0.59895831 },
    {     0.20992239,    -0.63713288,    -0.59895831 },
    {    -0.20992239,    -0.63713288,    -0.59895831 },
    {    -0.63713288,    -0.20992239,    -0.59895831 },
    {     0.84705532,     0.00000000,    -0.30208334 },
    {     0.97227210,     0.00000000,     0.50000000 },
    {     0.01104855,    -0.85810387,     0.31770834 },
    {     0.00000000,    -0.84705532,    -0.30208334 },
};

TEST_CASE("USD Catmull-Clark matches OpenSubdiv on catmark_cube_creases0", "[usd][subdiv]")
{
    using namespace Slic3r::usd_subdiv;

    const std::vector<Pt> points = {
        {  0.0,      -1.414214,  1.0 }, {  1.414214,  0.0,       1.0 },
        { -1.414214,  0.0,       1.0 }, {  0.0,       1.414214,  1.0 },
        { -1.414214,  0.0,      -1.0 }, {  0.0,       1.414214, -1.0 },
        {  0.0,      -1.414214, -1.0 }, {  1.414214,  0.0,      -1.0 },
    };
    const std::vector<int> counts  = { 4, 4, 4, 4, 4, 4 };
    const std::vector<int> indices = { 0,1,3,2,  2,3,5,4,  4,5,7,6,  6,7,1,0,  1,7,5,3,  6,0,2,4 };
    // The crease 0-1-3 as two USD length-2 chains, exercising build_cage's chain->edge expansion.
    const std::vector<int>    creaseIndices = { 0, 1,   1, 3 };
    const std::vector<int>    creaseLengths = { 2, 2 };
    const std::vector<double> creaseSharp   = { 2.0, 2.0 };
    const std::vector<int>    cornerIndices;
    const std::vector<double> cornerSharp;

    Cage cage = build_cage(points, counts, indices, creaseIndices, creaseLengths, creaseSharp,
                           cornerIndices, cornerSharp, BOUNDARY_EDGE_AND_CORNER, /*triangle_smooth=*/false);
    cage = subdivide(cage, "catmullClark", 2);

    REQUIRE(cage.verts.size() == 98);

    // The two number vertices differently, so match by nearest neighbour and require a bijection --
    // else two of ours could collapse onto one golden vertex and still pass.
    const double TOL2 = 1e-5 * 1e-5;
    std::array<int, 98> hit{};
    double worst2 = 0.0;
    for (const Pt &p : cage.verts) {
        int best = -1; double bestd2 = 1e30;
        for (int j = 0; j < 98; ++j) {
            const double dx = p[0] - CREASES0_GOLDEN_L2[j][0];
            const double dy = p[1] - CREASES0_GOLDEN_L2[j][1];
            const double dz = p[2] - CREASES0_GOLDEN_L2[j][2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bestd2) { bestd2 = d2; best = j; }
        }
        worst2 = std::max(worst2, bestd2);
        if (best >= 0) ++hit[best];   // best is -1 only on a NaN coordinate, which worst2 catches
    }
    INFO("worst nearest-neighbour distance to OpenSubdiv golden: " << std::sqrt(worst2));
    REQUIRE(worst2 < TOL2);
    REQUIRE(*std::max_element(hit.begin(), hit.end()) == 1);   // bijection: each golden hit once
}

SCENARIO_METHOD(UsdResourcesFixture, "A subdivision cage slices at slice time as it does eagerly",
                "[usd][subdiv][PrintMan]")
{
    GIVEN("cube_catmull imported eagerly (a flat mesh) and lazily (a cage scene)") {
        Model       eager, lazy;
        std::string msg;
        // amplify=false refines the cage at import (the reference); amplify=true defers it to a
        // one-placement PrintMan scene, refined by the slicer.
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &eager, msg, nullptr, false));
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &lazy,  msg, nullptr, true));

        const ModelVolume *ev = eager.objects.front()->volumes.front();
        const ModelVolume *lv = lazy.objects.front()->volumes.front();

        THEN("the lazy import defers a one-placement cage; the eager import is a flat mesh") {
            REQUIRE_FALSE(ev->printman_scene.has_value());
            REQUIRE(ev->mesh().facets_count() == 192);        // eager: subdivided at import (L2)
            REQUIRE(lv->printman_scene.has_value());
            REQUIRE(lv->printman_scene->prototypes.size() == 1);
            REQUIRE(lv->printman_scene->placements.size() == 1);
            REQUIRE(lv->printman_scene->cages.size() == 1);
            REQUIRE(lv->mesh().facets_count() == 12);         // proxy envelope is a single box
        }

        THEN("the proxy AABB is measured at the finest device level, so a device-perfect slice sits on the bed") {
            // The proxy AABB (read by arrange / ensure_on_bed) is the kDeviceLevelMax surface, not
            // level 2 -- else a fine slice contracts ~0.2 mm past it, floating the object off the bed.
            const PrintMan::SubdivCage &sc = lv->printman_scene->cages.begin()->second;
            const usd_subdiv::Cage cage = usd_subdiv::build_cage(
                sc.points, sc.face_counts, sc.face_indices, sc.crease_indices, sc.crease_lengths,
                sc.crease_sharp, sc.corner_indices, sc.corner_sharp, sc.boundary, sc.triangle_smooth);
            auto zspan = [&](int L, double &lo, double &hi) {
                const usd_subdiv::Cage r = usd_subdiv::subdivide(cage, sc.scheme, L);
                lo = 1e30; hi = -1e30;
                for (const usd_subdiv::Pt &p : r.verts) { lo = std::min(lo, p[2]); hi = std::max(hi, p[2]); }
            };
            double lo_fine, hi_fine, lo_two, hi_two;
            zspan(usd_subdiv::kDeviceLevelMax, lo_fine, hi_fine);   // the level the proxy is measured at
            zspan(usd_subdiv::kDeviceLevelMin, lo_two,  hi_two);    // the old proxy level
            REQUIRE((hi_two - lo_two) - (hi_fine - lo_fine) > 0.2); // the two levels genuinely differ

            // The recorded proxy AABB is the finest slice's, raised above the level-2 bottom that floated.
            REQUIRE(std::abs(sc.refined_lo[2] - lo_fine) < 0.02);
            REQUIRE(std::abs(sc.refined_hi[2] - hi_fine) < 0.02);
            REQUIRE(sc.refined_lo[2] > lo_two + 0.1);
        }

        THEN("both slice to the same contours, layer for layer") {
            // Slice both through slice_scene (no arrange to shift one against the other); the lazy
            // placement lands the refined surface at the eager mesh's world coords, so equal contours
            // prove slice-time == import-time refinement. Run twice, the second with a tilt that mixes
            // Z into X/Y to exercise the world-Z band keys. This cube refines to a single band.
            PrintMan::PrintManScene eager_scene;
            eager_scene.prototypes.push_back(ev->mesh().its);
            eager_scene.placements.push_back(PrintMan::Placement{});   // prototype 0, identity

            auto slice_and_compare = [&](const Transform3d &extra) {
                const Transform3d em = extra * ev->get_matrix();
                float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
                for (const Vec3f &v : ev->mesh().its.vertices) {
                    const float z = float((em * v.cast<double>()).z());
                    zmin = std::min(zmin, z);
                    zmax = std::max(zmax, z);
                }
                std::vector<float> zs;
                for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
                    zs.push_back(z);
                REQUIRE(zs.size() > 10);

                MeshSlicingParamsEx pe;  pe.trafo = em;
                MeshSlicingParamsEx pl;  pl.trafo = extra * lv->get_matrix();
                const std::vector<ExPolygons> ea = PrintMan::slice_scene(eager_scene, pe, zs);
                std::vector<std::pair<size_t, size_t>> prog;
                const std::vector<ExPolygons> la = PrintMan::slice_scene(*lv->printman_scene, pl, zs,
                    [](){}, [&](size_t d, size_t t){ prog.emplace_back(d, t); });
                // Progress reaches 100% and moves forward against a fixed total (this cube is a single band).
                REQUIRE(! prog.empty());
                REQUIRE(prog.back().first == prog.back().second);
                for (size_t i = 1; i < prog.size(); ++ i) {
                    REQUIRE(prog[i].second == prog[0].second);
                    REQUIRE(prog[i].first  >  prog[i - 1].first);
                }

                REQUIRE(ea.size() == zs.size());
                REQUIRE(la.size() == zs.size());
                double worst = 0.0, sum = 0.0;
                size_t worst_i = 0;
                for (size_t i = 0; i < zs.size(); ++ i) {
                    sum += total_area(ea[i]);
                    const double denom = std::max(total_area(ea[i]), total_area(la[i]));
                    if (denom <= 0.0)
                        continue;
                    const double rel = xor_area(ea[i], la[i]) / denom;
                    if (rel > worst) { worst = rel; worst_i = i; }
                }
                INFO("worst layer xor/area = " << worst << " @layer " << worst_i);
                REQUIRE(sum > 0.0);            // there is geometry to compare
                REQUIRE(worst < 1e-3);         // contours agree to <0.1% of layer area
            };

            slice_and_compare(Transform3d::Identity());
            slice_and_compare(Transform3d(Eigen::AngleAxisd(0.6, Vec3d(1.0, 0.4, 0.0).normalized())));
        }
    }
}

TEST_CASE("Subdivision slice progress advances per Z-band within one placement", "[usd][subdiv][PrintMan]")
{
    // A tall column of stacked box rings: each face's 1-ring spans only a couple of segments in Z, so
    // it refines across several Z-bands (unlike the cube's single band), and progress advances per band.
    const int    segments = 12;
    const double side     = 10.0;
    const double seg_h    = 2.0;

    PrintMan::SubdivCage cage;
    cage.scheme = "catmullClark";
    for (int k = 0; k <= segments; ++ k) {
        const double z = k * seg_h, h = side * 0.5;
        cage.points.push_back({{-h, -h, z}});
        cage.points.push_back({{ h, -h, z}});
        cage.points.push_back({{ h,  h, z}});
        cage.points.push_back({{-h,  h, z}});
    }
    auto quad = [&](int a, int b, int c, int d) {
        cage.face_counts.push_back(4);
        for (int i : {a, b, c, d}) cage.face_indices.push_back(i);
    };
    for (int k = 0; k < segments; ++ k)                             // side walls
        for (int e = 0; e < 4; ++ e)
            quad(4*k + e, 4*k + (e + 1) % 4, 4*(k + 1) + (e + 1) % 4, 4*(k + 1) + e);
    quad(0, 3, 2, 1);                                               // bottom cap (inward winding)
    quad(4*segments, 4*segments + 1, 4*segments + 2, 4*segments + 3);  // top cap

    PrintMan::PrintManScene scene;
    scene.prototypes.emplace_back();                     // cage slices from cages[0], not this mesh
    scene.placements.push_back(PrintMan::Placement{});   // prototype 0, identity
    scene.cages[0] = cage;

    std::vector<float> zs;
    for (float z = 0.1f; z < float(segments * seg_h) - 1e-4f; z += 0.2f)
        zs.push_back(z);
    REQUIRE(zs.size() > 50);

    MeshSlicingParamsEx params;
    // slice_scene serializes report_progress under a mutex, so appending here is safe under parallel bands.
    std::vector<std::pair<size_t, size_t>> prog;
    const std::vector<ExPolygons> la = PrintMan::slice_scene(scene, params, zs,
        [](){}, [&](size_t d, size_t t){ prog.emplace_back(d, t); });
    REQUIRE(la.size() == zs.size());
    REQUIRE(! prog.empty());
    const size_t bands = prog.front().second;
    INFO("bands = " << bands << ", progress calls = " << prog.size());
    REQUIRE(bands > 1);                                  // genuinely multi-band (not the cube's 1)
    // Parallel bands report out of order but exactly once each, so assert the set, not the sequence.
    REQUIRE(prog.size() == bands);                       // every band advanced the bar (report step is 1 here)
    std::vector<size_t> dones;
    for (const auto &pr : prog) { REQUIRE(pr.second == bands); dones.push_back(pr.first); }
    std::sort(dones.begin(), dones.end());
    for (size_t i = 0; i < dones.size(); ++ i)
        REQUIRE(dones[i] == i + 1);                      // exactly {1..bands}: each band bumped once, hit 100%
}

// The Z-band producer is Catmull-Clark only; a loop/bilinear cage must still slice via the whole-cage
// path, not silently to nothing. Bilinear's limit surface is the control cage itself.
SCENARIO_METHOD(UsdResourcesFixture, "A non-Catmull-Clark cage still slices, via the whole-cage path",
                "[usd][subdiv][PrintMan]")
{
    GIVEN("cube_bilinear imported eagerly (a flat mesh) and lazily (a cage scene)") {
        Model       eager, lazy;
        std::string msg;
        REQUIRE(load_usd(usd_path("cube_bilinear.usda").c_str(), &eager, msg, nullptr, false));
        REQUIRE(load_usd(usd_path("cube_bilinear.usda").c_str(), &lazy,  msg, nullptr, true));

        const ModelVolume *ev = eager.objects.front()->volumes.front();
        const ModelVolume *lv = lazy.objects.front()->volumes.front();

        THEN("the lazy import defers a bilinear cage, sliced through the plain path, not the band loop") {
            REQUIRE(lv->printman_scene.has_value());
            REQUIRE(lv->printman_scene->cages.size() == 1);
            REQUIRE(lv->printman_scene->cages.begin()->second.scheme == "bilinear");

            const Transform3d em = ev->get_matrix();
            float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
            for (const Vec3f &v : ev->mesh().its.vertices) {
                const float z = float((em * v.cast<double>()).z());
                zmin = std::min(zmin, z);
                zmax = std::max(zmax, z);
            }
            std::vector<float> zs;
            for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
                zs.push_back(z);
            REQUIRE(zs.size() > 10);

            PrintMan::PrintManScene eager_scene;
            eager_scene.prototypes.push_back(ev->mesh().its);
            eager_scene.placements.push_back(PrintMan::Placement{});

            MeshSlicingParamsEx pe;  pe.trafo = em;
            MeshSlicingParamsEx pl;  pl.trafo = lv->get_matrix();
            const std::vector<ExPolygons> ea = PrintMan::slice_scene(eager_scene, pe, zs);
            const std::vector<ExPolygons> la = PrintMan::slice_scene(*lv->printman_scene, pl, zs);

            double worst = 0.0, sum_lazy = 0.0;
            for (size_t i = 0; i < zs.size(); ++ i) {
                sum_lazy += total_area(la[i]);         // the regression: this was all-zero (cage vanished)
                const double denom = std::max(total_area(ea[i]), total_area(la[i]));
                if (denom <= 0.0)
                    continue;
                worst = std::max(worst, xor_area(ea[i], la[i]) / denom);
            }
            REQUIRE(sum_lazy > 0.0);       // the cage slices to something, not nothing
            REQUIRE(worst < 1e-3);         // and matches the eager whole-cage slice
        }
    }
}

// One cage prototype under N placements, refined per placement to the same part as the flattened
// eagerly-refined instances. One instance is rotated, exercising per-placement world-Z band selection
// through the importer. Compared via slice_scene in shared world coords (no arrange to shift them).
SCENARIO_METHOD(UsdResourcesFixture, "An instanced subdivision cage amplifies per placement",
                "[usd][subdiv][PrintMan]")
{
    GIVEN("instanced_catmull imported amplified (a cage scene) and naive (flattened refined instances)") {
        Model       ours, naive;
        std::string msg;
        REQUIRE(load_usd(usd_path("instanced_catmull.usda").c_str(), &ours,  msg, nullptr, true));
        REQUIRE(load_usd(usd_path("instanced_catmull.usda").c_str(), &naive, msg, nullptr, false));

        const ModelVolume *v = ours.objects.front()->volumes.front();

        THEN("the importer defers one cage prototype under three placements") {
            REQUIRE(v->printman_scene.has_value());
            REQUIRE(v->printman_scene->prototypes.size() == 1);
            REQUIRE(v->printman_scene->cages.size() == 1);              // the cage IS populated (the fix)
            REQUIRE(v->printman_scene->cages.begin()->second.scheme == "catmullClark");
            REQUIRE(v->printman_scene->placements.size() == 3);
        }

        THEN("it slices to the same contours as the flattened refined instances, layer for layer") {
            // Naive reference: the eagerly-refined instances baked to world and merged into one mesh.
            indexed_triangle_set merged;
            for (const ModelObject *o : naive.objects)
                for (const ModelVolume *nv : o->volumes) {
                    const Transform3d m = nv->get_matrix();
                    const int base = int(merged.vertices.size());
                    for (const Vec3f &p : nv->mesh().its.vertices)
                        merged.vertices.emplace_back((m * p.cast<double>()).cast<float>());
                    for (const stl_triangle_vertex_indices &t : nv->mesh().its.indices)
                        merged.indices.emplace_back(t[0] + base, t[1] + base, t[2] + base);
                }
            REQUIRE(merged.indices.size() > size_t(3 * 100));   // three refined cubes (192 tris each), not control cages (12)

            float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
            for (const Vec3f &p : merged.vertices) { zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z()); }
            std::vector<float> zs;
            for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.3f)
                zs.push_back(z);
            REQUIRE(zs.size() > 10);

            // Both sliced in the same world frame: the naive merge already world-space (identity
            // placement), the amplified scene un-centred by the volume's own transform.
            PrintMan::PrintManScene naive_scene;
            naive_scene.prototypes.push_back(merged);
            naive_scene.placements.push_back(PrintMan::Placement{});

            MeshSlicingParamsEx pn;  pn.trafo = Transform3d::Identity();
            MeshSlicingParamsEx po;  po.trafo = v->get_matrix();
            const std::vector<ExPolygons> na = PrintMan::slice_scene(naive_scene, pn, zs);
            const std::vector<ExPolygons> oa = PrintMan::slice_scene(*v->printman_scene, po, zs);

            double worst = 0.0, sum = 0.0;
            size_t worst_i = 0;
            for (size_t i = 0; i < zs.size(); ++ i) {
                sum += total_area(na[i]);
                const double denom = std::max(total_area(na[i]), total_area(oa[i]));
                if (denom <= 0.0)
                    continue;
                const double rel = xor_area(na[i], oa[i]) / denom;
                if (rel > worst) { worst = rel; worst_i = i; }
            }
            INFO("worst layer xor/area = " << worst << " @layer " << worst_i);
            REQUIRE(sum > 0.0);
            REQUIRE(worst < 1e-3);
        }
    }
}

// read_prototype_mesh enforces read_stage's guards: a prototype with a non-finite point or authored
// holeIndices is refused, not imported (else the NaN poisons the bounding box, the hole imports filled).
// A good prototype keeps the amplified scene in use, so read_stage's fall-through never runs here.
SCENARIO_METHOD(UsdResourcesFixture, "The instancing path refuses non-finite and holed prototypes",
                "[usd][PrintMan]")
{
    GIVEN("an instanced stage with a good, a NaN-point, and a holeIndices prototype") {
        Model       model;
        std::string msg;
        REQUIRE(load_usd(usd_path("instanced_bad_mixed.usda").c_str(), &model, msg, nullptr, true));

        const ModelVolume *v = model.objects.front()->volumes.front();
        REQUIRE(v->printman_scene.has_value());

        THEN("only the good prototype survives, and the bounding box stays finite") {
            REQUIRE(v->printman_scene->prototypes.size() == 1);   // NaN and holed prototypes refused
            REQUIRE(v->printman_scene->placements.size() == 1);   // only the good instance is placed
            const BoundingBoxf3 bb = model.objects.front()->bounding_box_exact();
            REQUIRE(bb.min.allFinite());                          // the NaN prototype did not poison bounds
            REQUIRE(bb.max.allFinite());
        }
    }
}

// The refinement level is chosen from the device rate (subdiv_tol), not fixed, so one cage slices at
// different densities. The band path must land on that level -- proven by matching a whole-cage
// refinement at the same level -- and a coarser rate must use fewer.
SCENARIO_METHOD(UsdResourcesFixture, "A subdivision cage slices at a device-perfect level",
                "[usd][subdiv][PrintMan]")
{
    GIVEN("cube_catmull imported as a lazy cage scene") {
        Model       lazy;
        std::string msg;
        REQUIRE(load_usd(usd_path("cube_catmull.usda").c_str(), &lazy, msg, nullptr, true));
        const ModelVolume *lv = lazy.objects.front()->volumes.front();
        REQUIRE(lv->printman_scene.has_value());
        const PrintMan::PrintManScene &scene = *lv->printman_scene;
        REQUIRE(scene.cages.size() == 1);
        REQUIRE(scene.placements.size() == 1);
        const PrintMan::SubdivCage &sc = scene.cages.begin()->second;

        // The engine's level inputs, computed here so the reference is built at the same level.
        const usd_subdiv::Cage cage = usd_subdiv::build_cage(
            sc.points, sc.face_counts, sc.face_indices, sc.crease_indices, sc.crease_lengths,
            sc.crease_sharp, sc.corner_indices, sc.corner_sharp, sc.boundary, sc.triangle_smooth);
        const Transform3d m       = lv->get_matrix() * scene.placements.front().xform;
        const double      sigma   = Eigen::JacobiSVD<Matrix3d>(Matrix3d(m.linear())).singularValues()(0);
        const double      medge   = usd_subdiv::max_control_edge(cage);
        const double      mcrease  = usd_subdiv::max_crease_sharpness(cage);
        auto level_for = [&](double tol) {
            return usd_subdiv::device_level(sigma, medge, tol, mcrease);
        };

        // Layer z's over the surface's world z-range.
        float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
        for (const usd_subdiv::Pt &p : usd_subdiv::subdivide(cage, sc.scheme, 2).verts) {
            const float z = float((m * Vec3d(p[0], p[1], p[2])).z());
            zmin = std::min(zmin, z); zmax = std::max(zmax, z);
        }
        std::vector<float> zs;
        for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f) zs.push_back(z);
        REQUIRE(zs.size() > 10);

        // The lazy cage sliced at a device rate (the band path picks the level from subdiv_tol).
        auto lazy_at = [&](double tol) {
            MeshSlicingParamsEx pl;  pl.trafo = lv->get_matrix();  pl.subdiv_tol = tol;
            return PrintMan::slice_scene(scene, pl, zs);
        };
        // The whole cage refined to a fixed level, sliced at the same placement -- the trusted reference.
        auto whole_at = [&](int L) {
            const usd_subdiv::Cage r = usd_subdiv::subdivide(cage, sc.scheme, L);
            indexed_triangle_set its;
            its.vertices.reserve(r.verts.size());
            for (const usd_subdiv::Pt &p : r.verts) its.vertices.emplace_back(float(p[0]), float(p[1]), float(p[2]));
            for (int f = 0; f < r.nfaces(); ++ f) {
                const int off = r.foff[f], n = r.foff[f + 1] - off;
                for (int k = 1; k + 1 < n; ++ k) {
                    int a = r.fvi[off], b = r.fvi[off + k], d = r.fvi[off + k + 1];
                    if (sc.flip_winding) std::swap(b, d);
                    its.indices.emplace_back(a, b, d);
                }
            }
            PrintMan::PrintManScene ref;
            ref.prototypes.push_back(its);
            ref.placements = scene.placements;   // same placement -> same world frame
            MeshSlicingParamsEx pr;  pr.trafo = lv->get_matrix();
            return PrintMan::slice_scene(ref, pr, zs);
        };
        auto worst_xor = [](const std::vector<ExPolygons> &a, const std::vector<ExPolygons> &b) {
            double w = 0.0;
            for (size_t i = 0; i < a.size(); ++ i) {
                const double denom = std::max(total_area(a[i]), total_area(b[i]));
                if (denom > 0.0) w = std::max(w, xor_area(a[i], b[i]) / denom);
            }
            return w;
        };

        THEN("a fine device rate refines past the old fixed 2, and the band path matches the whole cage at that level") {
            const int L = level_for(0.2);
            REQUIRE(L > 2);                                         // device-perfect actually engaged
            REQUIRE(worst_xor(lazy_at(0.2), whole_at(L)) < 1e-3);   // band == whole cage at the device level
            REQUIRE(worst_xor(lazy_at(0.2), whole_at(2)) > 1e-3);   // and NOT the old level 2 (the surface moved)
        }

        THEN("a coarser device rate uses fewer levels -- one cage, two densities") {
            REQUIRE(level_for(0.8) < level_for(0.2));               // responds to the device rate
            const int L = level_for(0.8);
            REQUIRE(L >= 2);
            REQUIRE(worst_xor(lazy_at(0.8), whole_at(L)) < 1e-3);
        }
    }
}

// An infinitely-sharp crease (USD sharpness 10) must NOT floor the level -- it stays sharp at every
// level, so counting it forced every hard-edged cage to the cap with a false faceting warning.
TEST_CASE("Device-perfect level excludes an infinitely-sharp crease from the floor", "[usd][subdiv]")
{
    using namespace Slic3r::usd_subdiv;
    const std::vector<Pt>  cube = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const std::vector<int> cc   = {4,4,4,4,4,4};
    const std::vector<int> ci   = {0,3,2,1, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7};

    const Cage inf  = build_cage(cube, cc, ci, {0,1}, {2}, {10.0}, {}, {}, BOUNDARY_EDGE_AND_CORNER, false);
    const Cage semi = build_cage(cube, cc, ci, {0,1}, {2}, {5.0},  {}, {}, BOUNDARY_EDGE_AND_CORNER, false);
    REQUIRE(max_crease_sharpness(inf)  == 0.0);   // s = 10 (infinite) excluded
    REQUIRE(max_crease_sharpness(semi) == 5.0);   // s = 5 (semi-sharp) floors the level
    bool capped = false;
    device_level(1.0, 1.0, 0.2, max_crease_sharpness(inf), &capped);
    REQUIRE_FALSE(capped);                        // the hard crease no longer trips the ceiling warning
}

// The limit AABB (for bed placement) must capture a feature frozen at its control position past where
// the rest converges, else the early-stop records a mis-placed extreme a finer slice then moves. The
// centre vertex is a corner of INTEGER sharpness 6: it unfreezes at 7, not ceil(6)=6 -- the off-by-one.
TEST_CASE("The limit AABB captures a semi-sharp feature frozen past convergence", "[usd][subdiv]")
{
    using namespace Slic3r::usd_subdiv;
    std::vector<Pt> gp;
    for (int y = 0; y < 5; ++ y) for (int x = 0; x < 5; ++ x) gp.push_back({double(x) - 2, double(y) - 2, 0.0});
    gp[12] = {0, 0, -3};
    std::vector<int> gc, gi;
    auto V = [](int x, int y) { return y * 5 + x; };
    for (int y = 0; y < 4; ++ y) for (int x = 0; x < 4; ++ x) {
        gc.push_back(4);
        gi.push_back(V(x, y)); gi.push_back(V(x + 1, y)); gi.push_back(V(x + 1, y + 1)); gi.push_back(V(x, y + 1));
    }
    const Cage cage = build_cage(gp, gc, gi, {}, {}, {}, {12}, {6.0}, BOUNDARY_EDGE_AND_CORNER, false);

    double lo8 = 1e30;
    for (const Pt &p : subdivide(cage, "catmullClark", kDeviceLevelMax).verts) lo8 = std::min(lo8, p[2]);
    REQUIRE(lo8 > -3.0 + 0.05);   // the apex genuinely unfreezes -- it rises above the control spike

    Pt lo, hi;
    refined_limit_aabb(cage, "catmullClark", kDeviceLevelMax, lo, hi);
    REQUIRE(std::abs(lo[2] - lo8) < 1e-3);   // the recorded AABB is the finest slice's, not the frozen -3
}

// Headless repro of the amplified-colour path (no GUI/OSL): an amplified scene declaring filaments must
// build one print region per filament at model-apply, the same way painted facets do.
SCENARIO_METHOD(UsdResourcesFixture, "PrintMan colour builds per-filament print regions", "[usd][printman][color]")
{
    ::setenv("PRINTMAN_DEBUG_COLOR", "1", 1);   // exactly what the GUI process carries
    Model       model;
    std::string message;
    REQUIRE(load_usd(usd_path("cube_grid.usda").c_str(), &model, message, nullptr, true));
    ModelVolume *vol = model.objects.front()->volumes.front();
    REQUIRE(vol->printman_scene.has_value());
    REQUIRE(vol->printman_scene->filaments.size() == 2);   // the importer must set this from the env var
    model.add_default_instances();
    model.center_instances_around_point(Vec2d(125.0, 125.0));   // on the bed so apply keeps the object

    DynamicPrintConfig config;
    config.apply(FullPrintConfig::defaults());
    config.set_key_value("filament_diameter", new ConfigOptionFloats(std::vector<double>{1.75, 1.75}));
    config.set_key_value("filament_colour",   new ConfigOptionStrings(std::vector<std::string>{"#FF0000", "#0000FF"}));
    config.set_key_value("single_extruder_multi_material", new ConfigOptionBool(true));   // like the Kobra X

    Print print;
    print.apply(model, config);
    REQUIRE(! print.objects().empty());

    const PrintObject *po = print.objects().front();
    for (size_t i = 0; i < po->num_printing_regions(); ++ i)
        WARN("region " << i << " outer_wall_filament_id=" << po->printing_region(i).config().outer_wall_filament_id.value);
    CHECK(po->num_printing_regions() > 1);   // base + one region per declared filament

    // Slice, and confirm apply_segmentation actually distributes contours to BOTH filament regions.
    try { print.process(); } catch (const std::exception &e) { WARN("process() threw: " << e.what()); }
    po = print.objects().front();
    std::vector<size_t> nonempty(po->num_printing_regions(), 0);
    for (const Layer *ly : po->layers())
        for (size_t i = 0; i < nonempty.size() && i < size_t(ly->region_count()); ++ i)
            if (! ly->get_region(int(i))->slices.empty()) ++ nonempty[i];
    for (size_t i = 0; i < nonempty.size(); ++ i)
        WARN("region " << i << " nonempty_layers=" << nonempty[i]);
    CHECK(nonempty[1] > 0);   // the upper-half filament-2 region must receive slices
}

// The GUI flow: the object is applied while the profile has one filament, then a second is added and the
// plate re-sliced. Print::apply is incremental, so the per-filament regions must regenerate on that change.
SCENARIO_METHOD(UsdResourcesFixture, "PrintMan colour regenerates regions when a filament is added", "[usd][printman][color]")
{
    ::setenv("PRINTMAN_DEBUG_COLOR", "1", 1);
    Model       model;
    std::string message;
    REQUIRE(load_usd(usd_path("cube_grid.usda").c_str(), &model, message, nullptr, true));
    model.add_default_instances();
    model.center_instances_around_point(Vec2d(125.0, 125.0));

    Print print;
    {   // first apply with ONE filament (import before a second is added)
        DynamicPrintConfig c; c.apply(FullPrintConfig::defaults());
        c.set_key_value("filament_diameter", new ConfigOptionFloats(std::vector<double>{1.75}));
        c.set_key_value("filament_colour",   new ConfigOptionStrings(std::vector<std::string>{"#FF0000"}));
        print.apply(model, c);
    }
    try { print.process(); } catch (const std::exception &) {}   // slice at 1 filament, like a GUI auto-slice
    {   // then re-apply with TWO (a filament was added, then re-slice)
        DynamicPrintConfig c; c.apply(FullPrintConfig::defaults());
        c.set_key_value("filament_diameter", new ConfigOptionFloats(std::vector<double>{1.75, 1.75}));
        c.set_key_value("filament_colour",   new ConfigOptionStrings(std::vector<std::string>{"#FF0000", "#0000FF"}));
        print.apply(model, c);
    }
    try { print.process(); } catch (const std::exception &) {}   // re-slice at 2 filaments (the GUI re-slice)
    REQUIRE(! print.objects().empty());
    const PrintObject *po = print.objects().front();
    std::vector<size_t> nonempty(po->num_printing_regions(), 0);
    for (const Layer *ly : po->layers())
        for (size_t i = 0; i < nonempty.size() && i < size_t(ly->region_count()); ++ i)
            if (! ly->get_region(int(i))->slices.empty()) ++ nonempty[i];
    for (size_t i = 0; i < nonempty.size(); ++ i)
        WARN("region " << i << " nonempty_layers=" << nonempty[i]);
    REQUIRE(po->num_printing_regions() > 1);
    CHECK(nonempty[1] > 0);   // the re-slice must fill the 2nd-filament region
}

#ifdef SLIC3R_OSL
// End-to-end through the REAL OSL gradient shader and the full process() pipeline (not just slice_scene):
// cube_gradient.usda declares printman_gradient + colorObjectSpace + colorDither, so the slicer feeds the
// shader an object-normalized Z and dithers magenta->cyan up the side. Filaments are set directly (NOT via
// PRINTMAN_DEBUG_COLOR), so the z-band fallback is unreachable -- only the real Cout path can colour this,
// and if the .oso fails to load the scene stays single-region and the test fails loudly. Asserts the
// gradient spans the object (a bottom-skewed region and a distinct top-skewed one) and that the dither
// puts BOTH filaments together across a wide middle band (a hard split could not).
SCENARIO_METHOD(UsdResourcesFixture, "The OSL gradient shader colours the whole object through process()", "[usd][printman][color][gradient][osl]")
{
    ::unsetenv("PRINTMAN_DEBUG_COLOR");   // force the real Cout path; no z-band fallback
    Model       model;
    std::string message;
    REQUIRE(load_usd(usd_path("cube_gradient.usda").c_str(), &model, message, nullptr, true));
    ModelVolume *vol = model.objects.front()->volumes.front();
    REQUIRE(vol->printman_scene.has_value());
    vol->printman_scene->filaments = {1, 2};                 // paint with the two loaded filaments
    REQUIRE(vol->printman_scene->color_object_space);        // the importer read the prim hints
    REQUIRE(vol->printman_scene->color_dither);
    model.add_default_instances();
    model.center_instances_around_point(Vec2d(125.0, 125.0));

    DynamicPrintConfig config;
    config.apply(FullPrintConfig::defaults());
    config.set_key_value("filament_diameter", new ConfigOptionFloats(std::vector<double>{1.75, 1.75}));
    config.set_key_value("filament_colour",   new ConfigOptionStrings(std::vector<std::string>{"#FF00FF", "#00FFFF"}));
    config.set_key_value("single_extruder_multi_material", new ConfigOptionBool(true));

    Print print;
    print.apply(model, config);
    REQUIRE(! print.objects().empty());
    try { print.process(); } catch (const std::exception &e) { WARN("process() threw: " << e.what()); }

    const PrintObject *po = print.objects().front();
    const size_t nR = po->num_printing_regions();
    // The real Cout shader split the object into filament regions (no fallback was available). The object's
    // default extruder IS filament 1, so painting filament 1 is a no-op and only the cyan (filament 2) walls
    // form an extra region: base+magenta = region 0, cyan = region 1.
    REQUIRE(nR == 2);

    const auto &layers = po->layers();
    const size_t nL = layers.size();
    REQUIRE(nL > 10);
    auto region_area = [](const Layer *ly, size_t i) {
        double a = 0.0;
        if (i < size_t(ly->region_count()))
            for (const Surface &s : ly->get_region(int(i))->slices.surfaces)
                a += s.expolygon.area();
        return a;
    };
    // Cyan (region 1) area per height-fifth: it should be sparse at the base and dense at the top -- the
    // object-space gradient carried all the way through process() into the sliced per-filament regions.
    const int NB = 5;
    double cyan[NB] = {0}, base[NB] = {0};
    for (size_t li = 0; li < nL; ++ li) {
        const int b = std::min(NB - 1, int(li * NB / nL));
        base[b] += region_area(layers[li], 0);
        cyan[b] += region_area(layers[li], 1);
    }
    for (int b = 0; b < NB; ++ b) {
        const double tot = base[b] + cyan[b];
        WARN("height fifth " << b << " cyan_area=" << cyan[b] << " cyan_share=" << (tot > 0 ? cyan[b] / tot : 0.0));
    }
    const double cyan_bottom = cyan[0] + cyan[1], cyan_top = cyan[NB - 2] + cyan[NB - 1];
    THEN("the cyan filament region is sparse at the base and dense at the top -- the gradient spans the object") {
        REQUIRE(cyan_bottom > 0.0);              // dither puts a little cyan even low down
        REQUIRE(cyan_top    > 0.0);
        REQUIRE(cyan_top > 2.0 * cyan_bottom);   // but it concentrates strongly toward the top of the object
    }
}

// End-to-end all-filament dithering through the REAL spectrum shader and the full process() pipeline.
// cube_spectrum.usda declares printman_spectrum + colorAllFilaments + colorObjectSpace + colorDither, so
// the slicer paints with EVERY loaded filament, dithering the hue sweep across the whole palette up the
// object. Six distinct filaments loaded; no PRINTMAN_DEBUG_COLOR, so only the real Cout path can colour it.
// Asserts the palette is genuinely distributed over the object's height: several filament regions receive
// area and their area-weighted mean heights span a wide range (a low colour and a high colour).
SCENARIO_METHOD(UsdResourcesFixture, "The spectrum shader dithers across all loaded filaments through process()", "[usd][printman][color][dither][allfilament][osl]")
{
    ::unsetenv("PRINTMAN_DEBUG_COLOR");
    Model       model;
    std::string message;
    REQUIRE(load_usd(usd_path("cube_spectrum.usda").c_str(), &model, message, nullptr, true));
    ModelVolume *vol = model.objects.front()->volumes.front();
    REQUIRE(vol->printman_scene.has_value());
    REQUIRE(vol->printman_scene->color_all_filaments);   // the importer read the prim hint
    REQUIRE(vol->printman_scene->filaments.empty());      // no explicit list -- resolved to all loaded
    model.add_default_instances();
    model.center_instances_around_point(Vec2d(125.0, 125.0));

    const std::vector<std::string> hexes = {"#FF0000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#FF00FF"};
    DynamicPrintConfig config;
    config.apply(FullPrintConfig::defaults());
    config.set_key_value("filament_diameter", new ConfigOptionFloats(std::vector<double>(hexes.size(), 1.75)));
    config.set_key_value("filament_colour",   new ConfigOptionStrings(hexes));
    config.set_key_value("single_extruder_multi_material", new ConfigOptionBool(true));

    Print print;
    print.apply(model, config);
    REQUIRE(! print.objects().empty());
    try { print.process(); } catch (const std::exception &e) { WARN("process() threw: " << e.what()); }

    const PrintObject *po = print.objects().front();
    const size_t nR = po->num_printing_regions();
    WARN("num_printing_regions = " << nR);
    REQUIRE(nR >= 4);   // the whole loaded palette split the object into many filament regions

    const auto &layers = po->layers();
    const size_t nL = layers.size();
    REQUIRE(nL > 10);
    auto region_area = [](const Layer *ly, size_t i) {
        double a = 0.0;
        if (i < size_t(ly->region_count()))
            for (const Surface &s : ly->get_region(int(i))->slices.surfaces)
                a += s.expolygon.area();
        return a;
    };
    // Per region: total area and area-weighted mean height (0=bottom, 1=top). A palette spread up the side
    // means the painted regions' mean heights range widely -- some colours low, some high.
    std::vector<double> asum(nR, 0.0), wsum(nR, 0.0);
    for (size_t li = 0; li < nL; ++ li) {
        const double t = double(li) / double(nL - 1);
        for (size_t i = 0; i < nR; ++ i) {
            const double a = region_area(layers[li], i);
            asum[i] += a; wsum[i] += a * t;
        }
    }
    size_t used = 0;
    double lo = 2.0, hi = -1.0;
    for (size_t i = 0; i < nR; ++ i) {
        const double mh = asum[i] > 0 ? wsum[i] / asum[i] : -1.0;
        WARN("region " << i << " area=" << asum[i] << " mean_height=" << mh);
        if (asum[i] > 0.0) ++ used;
        if (i >= 1 && asum[i] > 0.0) { lo = std::min(lo, mh); hi = std::max(hi, mh); }   // painted regions only
    }
    THEN("many filaments are used and their heights span the object (palette climbs the side)") {
        REQUIRE(used >= 4);        // at least four of the loaded filaments actually appear
        REQUIRE(hi - lo > 0.5);    // and the painted colours are spread from low on the object to high
    }
}

// THE faithful "what Karl sees" test: after the full process() pipeline, measure how much of the object's
// OUTER WALL (a one-perimeter ring of each layer's sliced surface -- not the infill-dominated whole region,
// not the pre-apply_segmentation ribbons) is left as the BASE filament vs claimed by the colour shader.
// Trick that makes it clean and discriminating: load the base filament (slot 1) as a neutral GREY the
// c->m->y->w sweep never emits -- so the base region is precisely the wall the colour FAILED to claim. (Grey,
// not white: the sweep ends at white at the top, so a white base would count the shader's own top-of-object
// white as uncovered and the metric could not tell the two apart.)
// A too-thin ribbon (the band_width bug) leaves the base grey showing over much of the wall; the fix
// leaves almost none. The base region is the object default (extruder 1) and is the pure complement of the
// coloured regions (apply_segmentation diffs only the base), so its ring share is the true uncovered fraction.
SCENARIO_METHOD(UsdResourcesFixture, "The surface colour claims the printed outer wall through process()", "[usd][printman][color][wallcolor][osl]")
{
    ::unsetenv("PRINTMAN_DEBUG_COLOR");
    Model       model;
    std::string message;
    REQUIRE(load_usd(usd_path("cube_spectrum.usda").c_str(), &model, message, nullptr, true));
    ModelVolume *vol = model.objects.front()->volumes.front();
    REQUIRE(vol->printman_scene.has_value());
    model.add_default_instances();
    model.center_instances_around_point(Vec2d(125.0, 125.0));

    // Slot 1 (the base/default extruder) = a neutral GREY the c->m->y->w sweep never emits; slots 2-5 are the
    // palette the sweep maps to (cyan, magenta, yellow, white). So region 0 (base=grey) is exactly the wall the
    // shader failed to colour. Grey, not white: the sweep ends at white, so a white base would count the shader's
    // own top-of-object white as "uncovered" and the metric could not tell the two apart.
    const std::vector<std::string> hexes = {"#808080", "#00FFFF", "#FF00FF", "#FFFF00", "#FFFFFF"};
    DynamicPrintConfig config;
    config.apply(FullPrintConfig::defaults());
    config.set_key_value("filament_diameter", new ConfigOptionFloats(std::vector<double>(hexes.size(), 1.75)));
    config.set_key_value("filament_colour",   new ConfigOptionStrings(hexes));
    config.set_key_value("single_extruder_multi_material", new ConfigOptionBool(true));

    Print print;
    print.apply(model, config);
    REQUIRE(! print.objects().empty());
    try { print.process(); } catch (const std::exception &e) { WARN("process() threw: " << e.what()); }

    const PrintObject *po = print.objects().front();
    const size_t nR = po->num_printing_regions();
    REQUIRE(nR >= 2);
    auto area = [](const ExPolygons &e) { double a = 0.0; for (const ExPolygon &p : e) a += p.area(); return a; };
    std::vector<double> wall(nR, 0.0);
    double ring_tot = 0.0;
    for (const Layer *ly : po->layers()) {
        if (ly->lslices.empty()) continue;
        const ExPolygons inner = offset_ex(ly->lslices, - float(scale_(0.45)));
        if (inner.empty()) continue;   // skip caps/thin layers where the one-perimeter ring is ill-defined
        const ExPolygons ring = diff_ex(ly->lslices, inner);
        ring_tot += area(ring);
        for (size_t i = 0; i < nR && i < size_t(ly->region_count()); ++ i) {
            ExPolygons rs;
            for (const Surface &s : ly->get_region(int(i))->slices.surfaces) rs.push_back(s.expolygon);
            wall[i] += area(intersection_ex(rs, ring));   // measured against the TRUE ring, not a double-counted sum
        }
    }
    REQUIRE(ring_tot > 0.0);
    const double base_share = wall[0] / ring_tot;   // region 0 = base (grey) = the uncovered wall fraction
    int coloured_present = 0;
    for (size_t i = 0; i < nR; ++ i) {
        const double sh = wall[i] / ring_tot;
        WARN("region " << i << " outer_wall_share=" << sh);
        if (i >= 1 && sh > 0.05) ++ coloured_present;   // filament i genuinely shows on the wall (>5%)
    }
    THEN("the colour claims almost all of the outer wall, spread across every filament (not left as base)") {
        REQUIRE(base_share < 0.15);       // the shader colours >= ~85% of the visible wall (the band_width fix)
        REQUIRE(coloured_present >= 4);    // and every filament in the c->m->y->w sweep genuinely shows on the wall
    }
}

// Headless RENDER of what the printed side looks like: run the exact GUI path (cube_spectrum's real .oso
// through process(), a grey base plus the c/m/y/w palette loaded -- 5 filaments), then for each layer
// bottom->top blend the outer-wall colour by each filament region's share of the ring, and dump it as a
// vertical colour strip to /tmp/colorviz.ppm. Looking at that image is the "test without the GUI" -- a clean
// bottom-to-top gradient means the engine is producing it; a flat bar would mean it is not. Also reports whether a wipe/prime
// tower is generated (the "cube in the corner"). Not an assertion -- a diagnostic dump I inspect.
SCENARIO_METHOD(UsdResourcesFixture, "Render the amplified surface colour to an image", "[usd][printman][color][colorviz][osl]")
{
    ::unsetenv("PRINTMAN_DEBUG_COLOR");
    Model       model;
    std::string message;
    REQUIRE(load_usd(usd_path("cube_spectrum.usda").c_str(), &model, message, nullptr, true));
    model.add_default_instances();
    model.center_instances_around_point(Vec2d(125.0, 125.0));

    // Slot 1 = grey base (a colour the c->m->y->w sweep never emits) so uncovered wall renders as grey and is
    // distinct from the gradient; slots 2-5 are the palette the sweep maps to. region i -> col[i].
    const std::vector<std::array<int,3>> col = {{128,128,128},{0,255,255},{255,0,255},{255,255,0},{255,255,255}};
    const std::vector<std::string> hexes = {"#808080", "#00FFFF", "#FF00FF", "#FFFF00", "#FFFFFF"};
    DynamicPrintConfig config;
    config.apply(FullPrintConfig::defaults());
    config.set_key_value("filament_diameter", new ConfigOptionFloats(std::vector<double>(hexes.size(), 1.75)));
    config.set_key_value("filament_colour",   new ConfigOptionStrings(hexes));
    config.set_key_value("single_extruder_multi_material", new ConfigOptionBool(true));

    Print print;
    print.apply(model, config);
    try { print.process(); } catch (const std::exception &e) { WARN("process() threw: " << e.what()); }

    // "Cube in the corner": is a wipe/prime tower being generated?
    WARN("enable_prime_tower=" << print.config().enable_prime_tower.value
         << " has_wipe_tower_data=" << (print.wipe_tower_data(int(print.config().filament_diameter.size())).depth > 0.f));

    const PrintObject *po = print.objects().front();
    const size_t nR = po->num_printing_regions();
    auto area = [](const ExPolygons &e) { double a = 0.0; for (const ExPolygon &p : e) a += p.area(); return a; };
    // Per layer bottom->top: area-weighted blend of the outer-wall colour.
    std::vector<std::array<int,3>> rows;
    for (const Layer *ly : po->layers()) {
        if (ly->lslices.empty()) { rows.push_back({40,40,40}); continue; }
        const ExPolygons inner = offset_ex(ly->lslices, - float(scale_(0.45)));
        const ExPolygons ring  = inner.empty() ? ly->lslices : diff_ex(ly->lslices, inner);
        double rt = 0.0, rr = 0.0, gg = 0.0, bb = 0.0;
        for (size_t i = 0; i < nR && i < size_t(ly->region_count()) && i < col.size(); ++ i) {
            ExPolygons rs; for (const Surface &s : ly->get_region(int(i))->slices.surfaces) rs.push_back(s.expolygon);
            const double a = area(intersection_ex(rs, ring));
            rt += a; rr += a * col[i][0]; gg += a * col[i][1]; bb += a * col[i][2];
        }
        if (rt > 0) rows.push_back({int(rr/rt), int(gg/rt), int(bb/rt)}); else rows.push_back({30,30,30});
    }
    // Write a PPM strip (bottom of the object at the image bottom), 120 px wide.
    if (FILE *f = std::fopen("/tmp/colorviz.ppm", "w")) {
        const int W = 120, H = int(rows.size());
        std::fprintf(f, "P3\n%d %d\n255\n", W, H);
        for (int y = 0; y < H; ++ y) {
            const auto &c = rows[H - 1 - y];   // top of image = top of object
            for (int x = 0; x < W; ++ x) std::fprintf(f, "%d %d %d ", c[0], c[1], c[2]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        WARN("wrote /tmp/colorviz.ppm (" << rows.size() << " layers)");
    }
    REQUIRE(! rows.empty());
}
#endif // SLIC3R_OSL
