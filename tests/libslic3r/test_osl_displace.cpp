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
#include <optional>
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
#include "libslic3r/PrintMan/Palette.hpp"
#include "libslic3r/Format/USDSubdiv.hpp"
#include <array>

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
    GIVEN("cube_grid.usda: a subdiv cube whose UsdShade material binds printman_grid to displacement") {
        Model model; std::vector<float> zs; MeshSlicingParamsEx params;
        const ModelVolume *vol = load_cage(model, "cube_grid.usda", zs, params);
        REQUIRE(zs.size() > 10);

        THEN("the importer resolved the displacement shader and its bound off the material") {
            REQUIRE(vol->printman_scene->osl_displacement_shader == "printman_grid");
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
            PrintMan::OslDisplaceShader osl(PRINTMAN_OSL_TEST_DIR, vol->printman_scene->osl_displacement_shader);
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

// Derivative propagation: OSL's filterwidth()/texture() band-limit using the sample footprint carried
// in sg.dPdx/dPdy, which the wrapper must feed (see docs/OSL-INTEGRATION.md). deriv_probe.osl outputs
// length(filterwidth(P)): a zero footprint -> ~0, a real footprint -> a matching positive width. Guards
// the antialiasing/texture work; needs deriv_probe.oso on the searchpath (built by CMake under SLIC3R_OSL).
SCENARIO("OSL displacement receives the sample footprint (derivatives)", "[osl]")
{
    OslDisplaceFixture fixture;
    GIVEN("deriv_probe.osl, which outputs the world-space filter footprint of P") {
        PrintMan::OslDisplaceShader probe(PRINTMAN_OSL_TEST_DIR, "deriv_probe", "Disp");
        const PrintMan::V3 p{{1.0, 2.0, 3.0}}, n{{0.0, 0.0, 1.0}};
        const PrintMan::V3 dPdx{{0.5, 0.0, 0.0}}, dPdy{{0.0, 0.5, 0.0}};
        THEN("no footprint -> ~0 out; a real footprint -> a positive width") {
            const double zero_fw = probe(p, n);              // 2-arg: dPdx/dPdy = 0
            const double real_fw = probe(p, n, dPdx, dPdy);  // ~0.5 in x and y
            INFO("zero_fw=" << zero_fw << "  real_fw=" << real_fw);
            REQUIRE(zero_fw == Catch::Approx(0.0).margin(1e-6));
            REQUIRE(real_fw > 0.1);
        }
    }
}

// Surface colour: a shader can return an `output color Cout` (linear RGB) as well as, or instead of,
// the float Disp -- both resolved from one shader, read per point. printman_test_color.osl sets
// Cout = color(P) and Disp = P[2], so a known point reads back a known colour. This is the M2 "shader
// colour out" step: it proves a surface shader's colour reaches the wrapper, feeding the (separately
// proven) per-filament sink. Needs printman_test_color.oso on the searchpath (built by CMake).
SCENARIO("OSL shader returns a surface colour (Cout)", "[osl][color]")
{
    OslDisplaceFixture fixture;
    GIVEN("printman_test_color.osl: Cout = color(P), Disp = P[2]") {
        PrintMan::OslDisplaceShader sh(PRINTMAN_OSL_TEST_DIR, "printman_test_color");
        const PrintMan::V3 n{{0.0, 0.0, 1.0}};
        THEN("it declares a colour output that reads back the shaded RGB per point") {
            REQUIRE(sh.has_color());
            const PrintMan::V3 c0 = sh.color(PrintMan::V3{{0.1, 0.2, 0.3}}, n);
            INFO("c0 = " << c0[0] << ", " << c0[1] << ", " << c0[2]);
            REQUIRE(c0[0] == Catch::Approx(0.1).margin(1e-5));
            REQUIRE(c0[1] == Catch::Approx(0.2).margin(1e-5));
            REQUIRE(c0[2] == Catch::Approx(0.3).margin(1e-5));
            // A different point gives a different colour -- P really drives the shader.
            const PrintMan::V3 c1 = sh.color(PrintMan::V3{{0.7, 0.4, 0.9}}, n);
            REQUIRE(c1[0] == Catch::Approx(0.7).margin(1e-5));
            // ...and the same shader's displacement output still reads (one eval, two outputs).
            REQUIRE(sh(PrintMan::V3{{0.1, 0.2, 0.3}}, n) == Catch::Approx(0.3).margin(1e-5));
        }
    }
    GIVEN("a displacement-only shader (printman_test_disp)") {
        PrintMan::OslDisplaceShader disp_only(PRINTMAN_OSL_TEST_DIR, "printman_test_disp");
        THEN("it reports no colour output, and color() is a benign zero") {
            REQUIRE_FALSE(disp_only.has_color());
            const PrintMan::V3 c = disp_only.color(PrintMan::V3{{1, 2, 3}}, PrintMan::V3{{0, 0, 1}});
            REQUIRE(c[0] == 0.0);
            REQUIRE(c[1] == 0.0);
            REQUIRE(c[2] == 0.0);
            REQUIRE(disp_only(PrintMan::V3{{1, 2, 3}}, PrintMan::V3{{0, 0, 1}}) > 0.0);
        }
    }
}

// The grid shader now also outputs a surface colour: red on the raised ribs (the visible lattice),
// blue in the grooves/flat surface -- "grid" vs "not-grid". Sample it over a lattice and assert Cout
// tracks Disp (a clearly-raised point is red, a clearly-cut point is blue) with no inversions and both
// colours occurring. This is the colour source the amplification path uses to drive filament choice.
SCENARIO("The grid shader colours grid vs not-grid by its displacement", "[osl][color][grid]")
{
    OslDisplaceFixture fixture;
    GIVEN("printman_grid with its Cout output") {
        PrintMan::OslDisplaceShader grid(PRINTMAN_OSL_TEST_DIR, "printman_grid");
        REQUIRE(grid.has_color());
        const PrintMan::V3 n{{0.5773, 0.5773, 0.5773}};   // generic normal: the pattern is not retired
        THEN("a clearly-raised sample is red, a clearly-cut sample is blue, and both occur") {
            size_t red_high = 0, blue_low = 0, violations = 0;
            for (double x = 0.0; x <= 3.0 + 1e-9; x += 0.5)
                for (double y = 0.0; y <= 3.0 + 1e-9; y += 0.5)
                    for (double z = 0.0; z <= 3.0 + 1e-9; z += 0.5) {
                        const PrintMan::V3 p{{x, y, z}};
                        const double       d = grid(p, n);          // Disp
                        const PrintMan::V3 c = grid.color(p, n);    // Cout
                        const bool is_red  = c[0] > 0.5;
                        const bool is_blue = c[2] > 0.5;
                        if (d > 0.6) { if (is_red) ++ red_high; else ++ violations; }   // clearly a rib
                        if (d < 0.1) { if (is_blue) ++ blue_low; else ++ violations; }  // clearly a groove
                    }
            INFO("red_high=" << red_high << " blue_low=" << blue_low << " violations=" << violations);
            REQUIRE(violations == 0);
            REQUIRE(red_high  > 0);
            REQUIRE(blue_low  > 0);
        }
    }
}

// Measurement (opt-in, hidden [.gridcolor]): slice the grid-shaded cage through the engine's colour
// path and report how the whole-layer (Phase A) vote distributes the two filaments -- so we can SEE
// whether the grid's within-layer colour survives whole-layer voting or collapses to one filament.
SCENARIO("Grid colour through the amplification engine -- per-layer distribution", "[.gridcolor]")
{
    OslDisplaceFixture fixture;
    Model model; std::vector<float> zs; MeshSlicingParamsEx params;
    const ModelVolume *vol = load_cage(model, "cube_grid.usda", zs, params);

    PrintMan::OslDisplaceShader osl(PRINTMAN_OSL_TEST_DIR, "printman_grid");
    REQUIRE(osl.has_color());

    PrintMan::DisplacementField disp;
    disp.max_magnitude = vol->printman_scene->osl_max_displacement;
    disp.eval_d = [&osl](const PrintMan::V3 &p, const PrintMan::V3 &n, const PrintMan::V3 &dx, const PrintMan::V3 &dy, double u, double v) { return osl(p, n, dx, dy, u, v); };

    PrintMan::ColorField color;
    color.palette = {FlushPredict::RGBColor(255, 0, 0), FlushPredict::RGBColor(0, 0, 255)};
    color.eval    = [&osl](const PrintMan::V3 &p, const PrintMan::V3 &n, const PrintMan::V3 &dx, const PrintMan::V3 &dy,
                           double u, double v) { return osl.color(p, n, dx, dy, u, v); };

    std::vector<std::vector<ExPolygons>> seg;
    const std::vector<ExPolygons>        layers =
        PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, disp, color, &seg);
    REQUIRE(seg.size() == zs.size());

    auto   area = [](const ExPolygons &e) { double a = 0.0; for (const ExPolygon &p : e) a += p.area(); return a; };
    size_t red = 0, blue = 0, both = 0, none = 0;
    for (size_t L = 0; L < seg.size(); ++ L) {
        const bool r = ! seg[L].empty() && area(seg[L][0]) > 0.0;
        const bool b = seg[L].size() > 1 && area(seg[L][1]) > 0.0;
        if (r) ++ red;
        if (b) ++ blue;
        if (r && b) ++ both;
        if (! r && ! b) ++ none;
    }
    // Phase B should show `both` high (grid rib+groove within the same layer); Phase A showed both=0.
    std::printf("[gridcolor] layers=%zu  red(rib)=%zu  blue(groove)=%zu  both=%zu  none=%zu\n", zs.size(), red, blue, both, none);
}

// Headless proof of the authored-UV texture pipeline end to end: load the sphere carrying face-varying
// primvars:st, refine it (Catmull-Clark -- the UV is refined face-varyingly ALONGSIDE position, the whole
// point of this feature), sample the bound earth shader at each refined face's UV (texture(earth.tx, u, v)),
// and render a front view of the coloured sphere to /tmp/earth_sphere.ppm. Recognisable continents prove the
// chain: authored st -> FVar refine -> shader u/v globals -> texture(). Hidden ([.earth]); a diagnostic image.
SCENARIO("earth texture maps onto the sphere through authored face-varying UV", "[.earth][osl]")
{
    Model model; std::vector<float> zs; MeshSlicingParamsEx params;
    const ModelVolume *vol = load_cage(model, "sphere_earth.usda", zs, params);
    REQUIRE(vol->printman_scene->cages.size() == 1);
    const PrintMan::SubdivCage &sc = vol->printman_scene->cages.begin()->second;
    REQUIRE(! sc.st.empty());   // the importer read the authored face-varying UV

    usd_subdiv::Cage cage = usd_subdiv::build_cage(
        sc.points, sc.face_counts, sc.face_indices, sc.crease_indices, sc.crease_lengths, sc.crease_sharp,
        sc.corner_indices, sc.corner_sharp, sc.boundary, sc.triangle_smooth, sc.st);
    REQUIRE(cage.has_uv());
    cage = usd_subdiv::subdivide(cage, "catmullClark", 5);
    REQUIRE(cage.has_uv());     // UV survived every refinement level

    // The earth shader + earth.tx are a local demo asset (not committed -- the texture is a 2.7 MB binary),
    // so skip gracefully if they are not staged; the [uvcolor] test covers the authored-UV path without them.
    std::optional<PrintMan::OslDisplaceShader> earth_opt;
    try { earth_opt.emplace(PRINTMAN_OSL_TEST_DIR, "printman_earth"); }
    catch (const std::exception &e) { WARN("skipping [.earth]: " << e.what()); return; }
    PrintMan::OslDisplaceShader &earth = *earth_opt;
    REQUIRE(earth.has_color());

    const int W = 220, H = 220;
    const double R = 15.5;
    std::vector<std::array<unsigned char, 3>> img(size_t(W) * H, {{20, 20, 30}});
    std::vector<double>                       depth(size_t(W) * H, 1e9);
    auto s2b = [](double x) { x = x < 0 ? 0 : (x > 1 ? 1 : x); return (unsigned char)(std::pow(x, 1.0 / 2.2) * 255.0 + 0.5); };
    // 8 cube-corner filaments (as the GUI would have loaded), and their sRGB bytes for the mosaic output.
    const std::vector<FlushPredict::RGBColor> palette = {
        {0,255,255},{255,0,255},{255,255,0},{255,255,255},{0,0,0},{0,255,0},{0,0,255},{255,0,0}};
    const int prgb[8][3] = {{0,255,255},{255,0,255},{255,255,0},{255,255,255},{0,0,0},{0,255,0},{0,0,255},{255,0,0}};
    int coloured = 0;
    for (int f = 0; f < cage.nfaces(); ++ f) {
        const int s = cage.foff[f], e = cage.foff[f + 1];
        usd_subdiv::Pt   c{{0, 0, 0}};
        std::array<double, 2> uv{{0, 0}};
        for (int k = s; k < e; ++ k) {
            const auto &p = cage.verts[cage.fvi[k]];
            c[0] += p[0]; c[1] += p[1]; c[2] += p[2];
            uv[0] += cage.fvar[k][0]; uv[1] += cage.fvar[k][1];
        }
        const double inv = 1.0 / (e - s);
        c[0] *= inv; c[1] *= inv; c[2] *= inv; uv[0] *= inv; uv[1] *= inv;
        if (c[1] > 0) continue;   // front hemisphere only (viewer at -Y looking +Y)
        const PrintMan::V3 col = earth.color(PrintMan::V3{{c[0], c[1], c[2]}}, PrintMan::V3{{0, -1, 0}},
                                             PrintMan::V3{{0, 0, 0}}, PrintMan::V3{{0, 0, 0}}, uv[0], uv[1]);
        const int px = int((c[0] + R) / (2 * R) * W), py = int((R - c[2]) / (2 * R) * H);   // z up -> image y down
        if (px < 0 || px >= W || py < 0 || py >= H) continue;
        const size_t idx = size_t(py) * W + px;
        if (c[1] < depth[idx]) {   // nearest front face wins
            depth[idx] = c[1];
            // The actual pipeline colour: dither this face's Cout to a filament, draw that filament's colour.
            const int k = PrintMan::dither_filament(col, palette, PrintMan::dither_hash(PrintMan::V3{{c[0],c[1],c[2]}}, 0.5));
            if (k >= 0) img[idx] = {{(unsigned char)prgb[k][0], (unsigned char)prgb[k][1], (unsigned char)prgb[k][2]}};
            else        img[idx] = {{s2b(col[0]), s2b(col[1]), s2b(col[2])}};
            (void)s2b;
            ++ coloured;
        }
    }
    if (FILE *fp = std::fopen("/tmp/earth_sphere.ppm", "w")) {
        std::fprintf(fp, "P3\n%d %d\n255\n", W, H);
        for (const auto &c : img) std::fprintf(fp, "%d %d %d ", c[0], c[1], c[2]);
        std::fclose(fp);
        WARN("wrote /tmp/earth_sphere.ppm (coloured " << coloured << " px from " << cage.nfaces() << " faces)");
    }
    REQUIRE(coloured > 500);
}

#endif // SLIC3R_OSL
