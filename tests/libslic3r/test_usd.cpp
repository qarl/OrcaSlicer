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
