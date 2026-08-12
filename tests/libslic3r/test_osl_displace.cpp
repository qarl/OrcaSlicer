// End-to-end: a user-authored OSL shader drives PrintMan displacement through slice_scene, the same
// path OrcaSlicer slices with -- proven by writing ONE displacement formula twice (OSL and C++) and
// asserting the engine produces the same slices from each, while both differ from the plain slice.
// This exercises the OslDisplaceShader under the engine's real tbb::parallel_for over bands.
//
// Built only when SLIC3R_OSL is ON (links a local liboslexec); the default build never sees it.

#ifdef SLIC3R_OSL

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "libslic3r/Model.hpp"
#include "libslic3r/Format/USD.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/TriangleMeshSlicer.hpp"
#include "libslic3r/PrintMan/Engine.hpp"
#include "libslic3r/PrintMan/OslShader.hpp"

using namespace Slic3r;

#ifndef PRINTMAN_OSL_TEST_DIR
#define PRINTMAN_OSL_TEST_DIR "."   // dir holding printman_test_disp.oso; set by CMake
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

// The exact model in printman_test_disp.osl (defaults Amp=0.3, Freq=3.0), as a C++ DisplaceShader.
static double test_formula(const PrintMan::V3 &p, const PrintMan::V3 &n)
{
    const double Amp = 0.3, Freq = 3.0;
    const double relief = 0.6 + 0.2 * std::sin(Freq * p[0]) + 0.2 * std::cos(Freq * p[1]);
    const double tilt   = 0.7 + 0.3 * std::fabs(n[2]);
    return Amp * relief * tilt;
}

SCENARIO_METHOD(OslDisplaceFixture, "An OSL shader drives displacement through slice_scene",
                "[usd][subdiv][PrintMan][displace][osl]")
{
    GIVEN("a subdivision cube and one displacement formula written as both OSL and C++") {
        Model       model;
        std::string message;
        REQUIRE(load_usd(osl_usd_path("cube_catmull.usda").c_str(), &model, message, nullptr, true));
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

        // max |disp| = Amp * 1.0 * 1.0 = 0.3 mm > the 0.2 mm layer height -> the seam-sound regime.
        const double max_mag = 0.3;

        PrintMan::DisplacementField cpp_disp;
        cpp_disp.max_magnitude = max_mag;
        cpp_disp.eval          = test_formula;

        PrintMan::OslDisplaceShader osl(PRINTMAN_OSL_TEST_DIR, "printman_test_disp", "Disp");
        PrintMan::DisplacementField osl_disp;
        osl_disp.max_magnitude = max_mag;
        osl_disp.eval          = [&osl](const PrintMan::V3 &p, const PrintMan::V3 &n) { return osl(p, n); };

        const std::vector<ExPolygons> plain = PrintMan::slice_scene(*vol->printman_scene, params, zs);
        const std::vector<ExPolygons> cpp_out =
            PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, cpp_disp);
        const std::vector<ExPolygons> osl_out =
            PrintMan::slice_scene(*vol->printman_scene, params, zs, [](){}, {}, osl_disp);

        REQUIRE(plain.size()   == zs.size());
        REQUIRE(cpp_out.size() == zs.size());
        REQUIRE(osl_out.size() == zs.size());

        auto total_area = [](const std::vector<ExPolygons> &layers) {
            double a = 0.0;
            for (const ExPolygons &layer : layers)
                for (const ExPolygon &ep : layer)
                    a += ep.area();
            return a;
        };
        const double a_plain = total_area(plain);
        const double a_cpp   = total_area(cpp_out);
        const double a_osl   = total_area(osl_out);

        THEN("the OSL shader really drove the engine (its slice differs from the plain one)") {
            REQUIRE(a_plain > 0.0);
            REQUIRE(a_osl   > 0.0);
            REQUIRE(std::abs(a_osl - a_plain) > 0.02 * a_plain);
        }
        THEN("the OSL slice matches the identical C++ shader, per layer and in total area") {
            REQUIRE(std::abs(a_osl - a_cpp) <= 1e-3 * a_cpp);
            for (size_t i = 0; i < zs.size(); ++i)
                REQUIRE(osl_out[i].size() == cpp_out[i].size());
        }
    }
}

#endif // SLIC3R_OSL
