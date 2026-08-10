// Integration test: a ModelVolume carrying a PrintManScene must slice through Orca's PrintObject
// pipeline as if its scene were baked. Discriminating by design: the envelope mesh is the scene's
// bounding BOX (one island) while the scene is four cube placements, so a four-island match can
// only mean the slice_volume branch fired and amplified the scene.

#include <catch2/catch_all.hpp>

#include "libslic3r/libslic3r.h"
#include "libslic3r/Model.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/Layer.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/Exception.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/PrintMan/PrintManScene.hpp"

#include "test_helpers.hpp"

#include <string>
#include <vector>

using namespace Slic3r;
using namespace Slic3r::Test;

namespace {

// Bake a scene into one mesh: transform each prototype by each placement, concatenate.
// The honest, memory-heavy reference the lazy engine must reproduce.
TriangleMesh bake(const PrintMan::PrintManScene &scene)
{
    indexed_triangle_set out;
    for (const PrintMan::Placement &pl : scene.placements) {
        const indexed_triangle_set &proto = scene.prototypes[pl.prototype];
        const int base = int(out.vertices.size());
        for (const auto &v : proto.vertices)
            out.vertices.push_back((pl.xform * v.cast<double>()).cast<float>());
        for (const auto &t : proto.indices)
            out.indices.push_back(stl_triangle_vertex_indices(t.x() + base, t.y() + base, t.z() + base));
    }
    return TriangleMesh(out);
}

// A solid axis-aligned box spanning [lo, hi] -- the conservative envelope.
indexed_triangle_set box_its(const Vec3f &lo, const Vec3f &hi)
{
    indexed_triangle_set b;
    b.vertices = {
        Vec3f(lo.x(), lo.y(), lo.z()), Vec3f(hi.x(), lo.y(), lo.z()),
        Vec3f(hi.x(), hi.y(), lo.z()), Vec3f(lo.x(), hi.y(), lo.z()),
        Vec3f(lo.x(), lo.y(), hi.z()), Vec3f(hi.x(), lo.y(), hi.z()),
        Vec3f(hi.x(), hi.y(), hi.z()), Vec3f(lo.x(), hi.y(), hi.z()),
    };
    const int faces[12][3] = {
        {0, 3, 2}, {0, 2, 1}, {4, 5, 6}, {4, 6, 7},
        {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5},
        {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7},
    };
    for (const auto &t : faces)
        b.indices.push_back(stl_triangle_vertex_indices(t[0], t[1], t[2]));
    return b;
}

PrintMan::Placement translate(int proto, double x, double y, double z)
{
    PrintMan::Placement pl;
    pl.prototype = proto;
    pl.xform     = Transform3d::Identity();
    pl.xform.translate(Vec3d(x, y, z));
    return pl;
}

// Total net area of a layer's islands (raw coord_t^2; compared only relatively).
double total_area(const Layer *layer)
{
    double a = 0.0;
    for (const ExPolygon &ep : layer->lslices)
        a += ep.area();
    return a;
}

// Slice `p`, returning "" on success or a message carrying the real inner SlicingError
// texts (Catch would otherwise surface only the "Errors" wrapper name).
std::string process_errors(Print &p)
{
    try { p.process(); return {}; }
    catch (const Slic3r::SlicingErrors &e) {
        std::string m = "SlicingErrors:";
        for (const auto &se : e.errors_) m += std::string(" [") + se.what() + "]";
        return m;
    }
    catch (const Slic3r::SlicingError &e) { return std::string("SlicingError: ") + e.what(); }
    catch (const std::exception &e)       { return std::string("exception: ") + e.what(); }
}

} // namespace

SCENARIO("A PrintMan volume slices through Orca's pipeline as if its scene were baked", "[PrintMan]")
{
    GIVEN("a scene of four disjoint cube placements (four islands)") {
        PrintMan::PrintManScene scene;
        scene.prototypes.push_back(Slic3r::Test::cube(10.0).its);
        scene.placements = {
            translate(0,  0.0, 0.0, 0.0),
            translate(0, 20.0, 0.0, 0.0),
            translate(0, 40.0, 0.0, 0.0),
            translate(0, 60.0, 0.0, 0.0),
        };

        const TriangleMesh       baked    = bake(scene);
        const BoundingBoxf3      bb       = baked.bounding_box();
        const TriangleMesh       envelope = TriangleMesh(box_its(bb.min.cast<float>(), bb.max.cast<float>()));
        const DynamicPrintConfig config   = DynamicPrintConfig::full_print_config();

        // Reference: slice the true baked geometry the normal (mesh) way.
        Print ref_print;
        Model ref_model;
        Slic3r::Test::init_print(std::vector<TriangleMesh>{ baked }, ref_print, ref_model, config);
        const std::string ref_err = process_errors(ref_print);
        INFO("reference process: " << ref_err);
        REQUIRE(ref_err.empty());

        // Lazy: a bounding-box envelope (one island, but contains the scene) carrying the
        // scene, so a four-island match can only come from the slice_volume branch running.
        Print pm_print;
        Model pm_model;
        Slic3r::Test::init_print(std::vector<TriangleMesh>{ envelope }, pm_print, pm_model, config);
        ModelVolume *vol = pm_model.objects.front()->volumes.front();

        // Express the scene in the volume's CENTERED frame. add_volume centres the envelope
        // and compensates in get_matrix(), so the scene must carry the opposite shift or
        // slice_volume's params.trafo * get_matrix() double-shifts it -- the same -mesh_offset
        // the USD importer applies at attach time.
        const Vec3d off = vol->source.mesh_offset;
        PrintMan::PrintManScene centered = scene;
        for (PrintMan::Placement &pl : centered.placements) {
            Transform3d s = Transform3d::Identity();
            s.translate(-off);
            pl.xform = s * pl.xform;
        }
        vol->printman_scene = centered;
        vol->set_new_unique_id();          // so Print::apply detects the change and reslices
        pm_print.apply(pm_model, config);
        const std::string pm_err = process_errors(pm_print);
        INFO("printman process: " << pm_err);
        REQUIRE(pm_err.empty());

        auto ref_layers = ref_print.objects().front()->layers();
        auto pm_layers  = pm_print.objects().front()->layers();

        THEN("the two agree layer for layer, in island count and area") {
            REQUIRE(pm_layers.size() == ref_layers.size());
            REQUIRE(pm_layers.size() > 0);
            bool any_material = false;
            for (size_t i = 0; i < ref_layers.size(); ++ i) {
                INFO("layer " << i);
                REQUIRE(pm_layers[i]->lslices.size() == ref_layers[i]->lslices.size());
                if (! ref_layers[i]->lslices.empty()) {
                    any_material = true;
                    REQUIRE_THAT(total_area(pm_layers[i]),
                                 Catch::Matchers::WithinRel(total_area(ref_layers[i]), 1e-3));
                }
            }
            REQUIRE(any_material);
        }

        THEN("a mid-height layer resolves to exactly four islands") {
            REQUIRE(pm_layers[pm_layers.size() / 2]->lslices.size() == 4);
        }
    }
}
