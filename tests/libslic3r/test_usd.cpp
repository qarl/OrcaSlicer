#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>

#include <catch2/catch_all.hpp>

#include "libslic3r/Model.hpp"
#include "libslic3r/Format/USD.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/Layer.hpp"
#include "libslic3r/ModelArrange.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/Exception.hpp"

using namespace Slic3r;

static inline std::string usd_path(const char *path)
{
    return std::string(TEST_DATA_DIR) + "/test_usd/" + path;
}

// Restored on the way out: resources_dir() is process-global and this binary is
// shared with every other suite.
struct UsdResourcesFixture
{
    UsdResourcesFixture() : previous(resources_dir()) { set_resources_dir(USD_PLUGIN_PARENT_DIR); }
    ~UsdResourcesFixture() { set_resources_dir(previous); }

    std::string previous;
};

SCENARIO_METHOD(UsdResourcesFixture, "Reading a USD file", "[usd]")
{
    GIVEN("a stage declaring metersPerUnit and a Y up-axis") {
        THEN("both are applied") {
            Model       model;
            std::string message;
            REQUIRE(load_usd(usd_path("cube_metres.usda").c_str(), &model, message));
            REQUIRE(is_approx(model.objects.front()->volumes.front()->mesh().size(),
                              Vec3d(10000, 10000, 10000)));

            // Position, not size: the inverse rotation gives identical extents
            // while placing the model under the plate.
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
}

// Every world-space vertex of a model, sorted, so two imports of the same stage are
// comparable regardless of how each centered its volumes. A PrintMan volume's geometry
// is prototype-through-placement; a flat volume's is its mesh.
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

// How many ORIENTED world triangles the two models share -- comparing connectivity + winding,
// which world_vertices()'s sorted vertex SET is blind to (that is the difference between "a
// different mesh" and "the same mesh sliced two ways"). Vertices are welded across both models
// by proximity, so the ~1e-6 mm path gap can't split a match while a tessellation edge keeps
// distinct vertices apart. Exact only for prototypes without a symmetry the readers can sample
// differently, so asserted on the committed fixtures and only logged for a PRINTMAN_USD override.
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
        // The same stage two ways. amplify=false is the trusted flatten path (native USD
        // transforms); amplify=true is the new scene path (manual row->column transpose).
        REQUIRE(load_usd(usd_path("instanced_transformed.usda").c_str(), &flat,  message, nullptr, false));
        REQUIRE(load_usd(usd_path("instanced_transformed.usda").c_str(), &scene, message, nullptr, true));

        THEN("the scene has one deduped prototype and three placements") {
            REQUIRE(scene.objects.front()->volumes.size() == 1);
            const ModelVolume *vol = scene.objects.front()->volumes.front();
            REQUIRE(vol->printman_scene.has_value());
            REQUIRE(vol->printman_scene->prototypes.size() == 1);
            REQUIRE(vol->printman_scene->placements.size() == 3);
            REQUIRE(flat.objects.front()->volumes.size() == 3);   // flatten = one per instance
            // The proxy is one 12-facet box PER INSTANCE, never the real geometry. For this cube
            // the box happens to equal the bake, so it is the world-vertex check below -- not
            // this count -- that proves the sliced geometry is the real thing.
            REQUIRE(vol->mesh().facets_count() == 12 * vol->printman_scene->placements.size());
        }

        THEN("scene and flatten place identical world geometry (so the transpose is right)") {
            // A wrong row-vector -> column-vector transpose is a no-op on translations but
            // diverges on the rotated/scaled instances, so this equality is the real gate.
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

// Area of the symmetric difference of two per-layer regions: (a \ b) + (b \ a). Zero iff the
// two regions are identical -- the region-equivalence metric, applied between the two Orca paths.
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
    // Deterministic, identical placement for both models (not arrange_objects, which can ROTATE
    // to pack -- and the proxy and flattened models have different 2D hulls, so it could orient
    // them differently and make the contour comparison meaningless).
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

// Correctness counterpart to the memory benchmark: the amplified path and the naive flatten
// must slice the SAME file to the SAME part. Both go through Orca's pipeline and their contours
// are compared layer for layer by symmetric-difference area. Fixtures: instanced_transformed
// (rotated + tilted/scaled instances, so a wrong per-placement transform diverges here) and
// torus_vertical (a hole-bearing, tilted, welded torus exercising the bake). Override with
// PRINTMAN_USD=<file> to prove it on any instanced stage.
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

            // Same geometry two ways => same object bounding box. If this ever fails, arrange
            // would place them differently and the contour comparison would be meaningless.
            const BoundingBoxf3 bb_o = ours.objects.front()->bounding_box_exact();
            const BoundingBoxf3 bb_n = naive.objects.front()->bounding_box_exact();
            REQUIRE((bb_o.min - bb_n.min).cwiseAbs().maxCoeff() < 1e-3);
            REQUIRE((bb_o.max - bb_n.max).cwiseAbs().maxCoeff() < 1e-3);

            // Do the two paths even present the SAME world geometry to the slicer? If this
            // diverges, the bug is in the reader/transform; if it's ~0, it's in the slicing.
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

            // ...and the SAME triangles (connectivity + winding), which the sorted vertex set
            // cannot see. Asserted on the committed fixtures; only logged for a PRINTMAN_USD
            // override, where a symmetric prototype may be re-sampled (see world_triangle_diff).
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
