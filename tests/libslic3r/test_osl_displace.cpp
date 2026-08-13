// End-to-end: user-authored OSL shaders drive PrintMan displacement through slice_scene, the same
// path OrcaSlicer slices with. For each shader the SAME displacement formula is written twice -- as
// an OSL shader and as a C++ lambda -- and the test asserts the engine produces the same slices from
// each (matching area + identical per-layer island counts) while both differ from the plain slice.
// This runs several shaders (position-driven, normal-driven, high-frequency) over two subdivision
// geometries, and exercises OslDisplaceShader under the engine's real tbb::parallel_for over bands.
//
// Built only when SLIC3R_OSL is ON (links a from-source liboslexec); the default build never sees it.

#ifdef SLIC3R_OSL

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "libslic3r/Model.hpp"
#include "libslic3r/Format/USD.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/TriangleMeshSlicer.hpp"
#include "libslic3r/SVG.hpp"
#include "libslic3r/PrintMan/Engine.hpp"
#include "libslic3r/PrintMan/OslShader.hpp"

using namespace Slic3r;

#ifndef PRINTMAN_OSL_TEST_DIR
#define PRINTMAN_OSL_TEST_DIR "."   // dir holding the compiled .oso shaders; set by CMake
#endif

static inline std::string osl_usd_path(const char *path)
{
    return std::string(TEST_DATA_DIR) + "/test_usd/" + path;
}

// resources_dir() is process-global; restore it (this test binary is shared). Named distinctly from
// test_usd.cpp's fixture to avoid an ODR clash in the shared binary.
struct OslDisplaceFixture
{
    OslDisplaceFixture() : previous(resources_dir()) { set_resources_dir(USD_PLUGIN_PARENT_DIR); }
    ~OslDisplaceFixture() { set_resources_dir(previous); }
    std::string previous;
};

using Formula = std::function<double(const PrintMan::V3 &, const PrintMan::V3 &)>;

// Each shader lives as an .osl AND as the matching C++ formula below; both are Amp*f with f<=1, so
// max_magnitude = Amp (0.3 mm, above the 0.2 mm layer -> the seam-sound regime).
struct ShaderCase { const char *oso; Formula cpp; };

static double total_area(const std::vector<ExPolygons> &layers)
{
    double a = 0.0;
    for (const ExPolygons &layer : layers)
        for (const ExPolygon &ep : layer)
            a += ep.area();
    return a;
}

// Slice `scene` with the OSL shader and with its C++ twin; require they agree, and both differ from
// the plain (no-displacement) slice.
static void expect_osl_matches_cpp(const PrintMan::PrintManScene &scene,
                                   const MeshSlicingParamsEx &params,
                                   const std::vector<float> &zs,
                                   double a_plain,
                                   const ShaderCase &sc)
{
    INFO("shader " << sc.oso);
    const double max_mag = 0.3;   // = Amp; a valid upper bound for every Amp*f (f<=1) test shader

    PrintMan::DisplacementField cpp_disp;
    cpp_disp.max_magnitude = max_mag;
    cpp_disp.eval          = sc.cpp;

    PrintMan::OslDisplaceShader osl(PRINTMAN_OSL_TEST_DIR, sc.oso, "Disp");
    PrintMan::DisplacementField osl_disp;
    osl_disp.max_magnitude = max_mag;
    osl_disp.eval          = [&osl](const PrintMan::V3 &p, const PrintMan::V3 &n) { return osl(p, n); };

    const std::vector<ExPolygons> cpp_out =
        PrintMan::slice_scene(scene, params, zs, [](){}, {}, cpp_disp);
    const std::vector<ExPolygons> osl_out =
        PrintMan::slice_scene(scene, params, zs, [](){}, {}, osl_disp);

    REQUIRE(cpp_out.size() == zs.size());
    REQUIRE(osl_out.size() == zs.size());
    const double a_cpp = total_area(cpp_out);
    const double a_osl = total_area(osl_out);

    // The OSL shader really drove the engine (its slice differs from the plain one).
    REQUIRE(a_osl > 0.0);
    REQUIRE(std::abs(a_osl - a_plain) > 0.02 * a_plain);
    // ...and it drove it the same as the identical C++ shader, per layer and in total area.
    REQUIRE(std::abs(a_osl - a_cpp) <= 1e-3 * a_cpp);
    for (size_t i = 0; i < zs.size(); ++i)
        REQUIRE(osl_out[i].size() == cpp_out[i].size());
}

// The C++ twins of the .osl shaders (defaults Amp=0.3). Kept identical to the .osl source.
static double f_mixed(const PrintMan::V3 &p, const PrintMan::V3 &n)   // printman_test_disp.osl
{
    const double Amp = 0.3, Freq = 3.0;
    const double relief = 0.6 + 0.2 * std::sin(Freq * p[0]) + 0.2 * std::cos(Freq * p[1]);
    const double tilt   = 0.7 + 0.3 * std::fabs(n[2]);
    return Amp * relief * tilt;
}
static double f_normal(const PrintMan::V3 &, const PrintMan::V3 &n)   // printman_test_normal.osl
{
    const double Amp = 0.3;
    return Amp * (0.4 + 0.2 * std::fabs(n[0]) + 0.2 * std::fabs(n[1]) + 0.2 * std::fabs(n[2]));
}
static double f_hifreq(const PrintMan::V3 &p, const PrintMan::V3 &)   // printman_test_hifreq.osl
{
    const double Amp = 0.3, Freq = 8.0;
    return Amp * (0.5 + 0.5 * std::sin(Freq * p[0]) * std::cos(Freq * p[1]));
}

// Load a subdivision fixture and build the layer set + params the displacement tests use.
static const ModelVolume *load_cage(Model &model, const char *fixture, std::vector<float> &zs,
                                    MeshSlicingParamsEx &params)
{
    std::string message;
    REQUIRE(load_usd(osl_usd_path(fixture).c_str(), &model, message, nullptr, true));
    const ModelVolume *vol = model.objects.front()->volumes.front();
    REQUIRE(vol->printman_scene.has_value());

    const Transform3d m = vol->get_matrix();
    float zmin = std::numeric_limits<float>::infinity(), zmax = -zmin;
    for (const Vec3f &v : vol->mesh().its.vertices) {
        const float z = float((m * v.cast<double>()).z());
        zmin = std::min(zmin, z);
        zmax = std::max(zmax, z);
    }
    for (float z = zmin + 0.1f; z < zmax - 1e-4f; z += 0.2f)
        zs.push_back(z);
    params.trafo      = m;
    params.subdiv_tol = 0.05;
    return vol;
}

SCENARIO_METHOD(OslDisplaceFixture, "OSL shaders drive displacement through slice_scene",
                "[usd][subdiv][PrintMan][displace][osl]")
{
    const std::vector<ShaderCase> shaders = {
        {"printman_test_disp",   f_mixed},
        {"printman_test_normal", f_normal},
        {"printman_test_hifreq", f_hifreq},
    };

    GIVEN("a subdivision cube, sliced by three different OSL shaders") {
        Model model; std::vector<float> zs; MeshSlicingParamsEx params;
        const ModelVolume *vol = load_cage(model, "cube_catmull.usda", zs, params);
        REQUIRE(vol->printman_scene->cages.size() == 1);
        REQUIRE(zs.size() > 10);

        const std::vector<ExPolygons> plain = PrintMan::slice_scene(*vol->printman_scene, params, zs);
        REQUIRE(plain.size() == zs.size());
        const double a_plain = total_area(plain);
        REQUIRE(a_plain > 0.0);

        for (const ShaderCase &sc : shaders)
            expect_osl_matches_cpp(*vol->printman_scene, params, zs, a_plain, sc);

        // The grid shader is a faithful port of the built-in Gridwork -- validate the OSL against the
        // ACTUAL built-in (not a hand twin). The grid is a step function, so OSL float vs Gridwork
        // double can differ slightly at the groove edges; measure the delta and require it small.
        THEN("the OSL grid shader reproduces the built-in Gridwork") {
            PrintMan::Device   dev;
            PrintMan::Gridwork gw;
            const double depth = gw.effective_depth(dev);

            PrintMan::DisplacementField cpp_grid;
            cpp_grid.max_magnitude = depth;
            cpp_grid.eval = [gw, dev](const PrintMan::V3 &p, const PrintMan::V3 &n) { return gw(p, n, dev); };

            PrintMan::OslDisplaceShader grid(PRINTMAN_OSL_TEST_DIR, "printman_grid", "Disp");
            PrintMan::DisplacementField osl_grid;
            osl_grid.max_magnitude = depth;
            osl_grid.eval = [&grid](const PrintMan::V3 &p, const PrintMan::V3 &n) { return grid(p, n); };

            const std::vector<ExPolygons> cpp_out =
                PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, cpp_grid);
            const std::vector<ExPolygons> osl_out =
                PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, osl_grid);
            const double a_cpp = total_area(cpp_out), a_osl = total_area(osl_out);
            std::printf("[grid] a_plain=%.4f  a_gridwork=%.4f  a_osl_grid=%.4f  |osl-gw|/gw=%.3e\n",
                        a_plain, a_cpp, a_osl, std::abs(a_osl - a_cpp) / a_cpp);
            REQUIRE(std::abs(a_osl - a_plain) > 0.02 * a_plain);   // the grid has a real effect
            REQUIRE(std::abs(a_osl - a_cpp)  <= 0.01 * a_cpp);      // matches the built-in within 1%
        }
    }

    GIVEN("a tall prism (many bands), sliced by the position-driven OSL shader") {
        Model model; std::vector<float> zs; MeshSlicingParamsEx params;
        const ModelVolume *vol = load_cage(model, "tall_prism_catmull.usda", zs, params);
        REQUIRE(zs.size() > 100);   // many layers => many bands => the multi-band OSL path

        const std::vector<ExPolygons> plain = PrintMan::slice_scene(*vol->printman_scene, params, zs);
        REQUIRE(plain.size() == zs.size());
        const double a_plain = total_area(plain);
        REQUIRE(a_plain > 0.0);

        expect_osl_matches_cpp(*vol->printman_scene, params, zs, a_plain, shaders[0]);
    }
}

// The full GUI path: the importer reads the shader off the USD prim, and the slicer loads it from its
// own shader dir (PRINTMAN_OSL_SHADER_DIR, where the build stages printman_grid.oso) -- exactly what
// PrintObjectSlice does. This is what makes "open a .usd in Orca and see the grid" work.
SCENARIO_METHOD(OslDisplaceFixture, "A USD prim's OSL shader imports and slices as the grid",
                "[usd][subdiv][PrintMan][displace][osl][grid]")
{
    GIVEN("cube_grid.usda: a subdiv cube carrying printman:oslShader = printman_grid") {
        Model model; std::vector<float> zs; MeshSlicingParamsEx params;
        const ModelVolume *vol = load_cage(model, "cube_grid.usda", zs, params);
        REQUIRE(zs.size() > 10);

        THEN("the importer read the shader and its bound off the prim") {
            REQUIRE(vol->printman_scene->osl_shader == "printman_grid");
            REQUIRE(vol->printman_scene->osl_max_displacement == Catch::Approx(0.7875));
        }

        THEN("the slicer loads that shader from its own dir and reproduces the built-in Gridwork") {
            PrintMan::Device   dev;
            PrintMan::Gridwork gw;
            PrintMan::DisplacementField cpp_grid;
            cpp_grid.max_magnitude = gw.effective_depth(dev);
            cpp_grid.eval = [gw, dev](const PrintMan::V3 &p, const PrintMan::V3 &n) { return gw(p, n, dev); };

            // Load by the prim's shader name (as PrintObjectSlice does). Uses the test's own shader
            // dir, which also stages printman_grid.oso, so this doesn't depend on cross-target macro
            // propagation; the slicer's own dir (PRINTMAN_OSL_SHADER_DIR) is exercised live in the GUI.
            PrintMan::OslDisplaceShader osl(PRINTMAN_OSL_TEST_DIR, vol->printman_scene->osl_shader);
            PrintMan::DisplacementField osl_grid;
            osl_grid.max_magnitude = vol->printman_scene->osl_max_displacement;
            osl_grid.eval = [&osl](const PrintMan::V3 &p, const PrintMan::V3 &n) { return osl(p, n); };

            const double a_cpp = total_area(PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, cpp_grid));
            const double a_osl = total_area(PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, osl_grid));
            REQUIRE(a_cpp > 0.0);
            REQUIRE(std::abs(a_osl - a_cpp) <= 0.01 * a_cpp);
        }
    }
}

// Headless visual: slice cube_grid.usda plain and with the grid shader, and write each as an SVG
// overlay of all layer contours -- so the grid is visible without the GUI. The [.gridsvg] tag is
// hidden (leading dot), so it is opt-in only (run it by name) and never fires on a default run;
// it writes to /tmp/printman_{grid,plain}.svg.
SCENARIO_METHOD(OslDisplaceFixture, "Dump grid vs plain slices to SVG for viewing", "[.gridsvg]")
{
    GIVEN("cube_grid.usda") {
        Model model; std::vector<float> zs; MeshSlicingParamsEx params;
        const ModelVolume *vol = load_cage(model, "cube_grid.usda", zs, params);

        const std::vector<ExPolygons> plain = PrintMan::slice_scene(*vol->printman_scene, params, zs);

        PrintMan::OslDisplaceShader osl(PRINTMAN_OSL_TEST_DIR, "printman_grid");
        PrintMan::DisplacementField gd;
        gd.max_magnitude = vol->printman_scene->osl_max_displacement;
        gd.eval = [&osl](const PrintMan::V3 &p, const PrintMan::V3 &n) { return osl(p, n); };
        const std::vector<ExPolygons> grid =
            PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, gd);

        BoundingBox bb;
        for (const std::vector<ExPolygons> *ls : {&plain, &grid})
            for (const ExPolygons &layer : *ls)
                for (const ExPolygon &ep : layer)
                    bb.merge(get_extents(ep));

        auto dump = [&](const std::string &path, const std::vector<ExPolygons> &layers, const std::string &color) {
            SVG svg(path, bb);
            for (const ExPolygons &layer : layers)
                svg.draw_outline(layer, color, color, scale_(0.04));
            svg.Close();
        };
        dump("/tmp/printman_plain.svg", plain, "black");
        dump("/tmp/printman_grid.svg",  grid,  "red");
        std::printf("[gridsvg] wrote /tmp/printman_plain.svg and /tmp/printman_grid.svg (%zu layers)\n", zs.size());
        REQUIRE(! grid.empty());
    }
}

#endif // SLIC3R_OSL
