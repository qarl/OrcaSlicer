#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/sdf/zipFile.h>          // SdfZipFile: read the .oso/.tx bundled in a self-contained .usdz
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

#include "libslic3r/Model.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/PrintMan/PrintManScene.hpp"
#include "libslic3r/Format/USDSubdiv.hpp"
#include "USD.hpp"

#ifdef _WIN32
#define DIR_SEPARATOR '\\'
#else
#define DIR_SEPARATOR '/'
#endif

PXR_NAMESPACE_USING_DIRECTIVE

namespace Slic3r {

namespace {

// A stage declares its own unit. metersPerUnit=1 means one unit is one metre,
// so geometry scales by 1000 into millimetres. Getting this wrong is not subtle
// -- it is a model 1000x too big -- but it is silent, so it is done here once
// and logged.
double mm_per_unit(const UsdStageRefPtr &stage)
{
    const double meters = UsdGeomGetStageMetersPerUnit(stage);
    return (meters > 0.0 ? meters : 0.01) * 1000.0;
}

// USD finds its file-format plugins (usda, usdc, usdz) and its asset resolver
// through plugInfo.json, which it searches for next to the *library* that
// contains it. We link USD statically into the OrcaSlicer binary, so that
// search looks beside the executable and finds nothing -- and the failure is a
// fatal abort inside UsdStage::Open, not a null return we could report.
//
// Registering explicitly keeps the app relocatable. PXR_INSTALL_LOCATION would
// also work but bakes an absolute build-machine path into the library.
//
// Returns false rather than latching, and says why: USD's reaction to a missing
// asset resolver is a fatal abort inside UsdStage::Open, so a quiet no-op here
// becomes a crash with no explanation. Not latching also lets an embedder that
// has not yet called set_resources_dir() succeed on a later attempt.
bool register_usd_plugins()
{
    namespace fs = boost::filesystem;

    static bool registered = false;
    if (registered)
        return true;

    // Beside the binary covers the macOS bundle and build-tree runs; resources/
    // covers the install() and AppImage layouts. resources_dir() can be empty in a test
    // binary or an embedder, which is why it is not the only candidate.
    std::vector<fs::path> candidates;
    candidates.emplace_back(boost::dll::program_location().parent_path() / "usd");
    if (!resources_dir().empty())
        candidates.emplace_back(fs::path(resources_dir()) / "usd");

    for (const fs::path &dir : candidates) {
        if (!fs::is_directory(dir))
            continue;
        if (!PlugRegistry::GetInstance().RegisterPlugins(dir.string()).empty()) {
            registered = true;
            return true;
        }
    }

    std::string tried;
    for (const fs::path &dir : candidates)
        tried += (tried.empty() ? "" : ", ") + dir.string();
    BOOST_LOG_TRIVIAL(error)
        << "load_usd: OpenUSD's plugInfo.json tree was not found (looked in: "
        << (tried.empty() ? "nowhere -- no candidate paths" : tried)
        << "). USD files cannot be read without it.";
    return false;
}

// Read at the earliest available time sample, NOT at Default().
//
// Usd.TimeCode.Default() never consults time samples. Houdini's USD ROP writes
// `points` as a single sample with no default value, so a stage that looks
// perfectly static reads back as zero meshes and the importer reports an empty
// file. EarliestTime() resolves both the sampled and the defaulted case.
// The exception is a stage that cannot say how fast its samples run. Resolving a
// transform at EarliestTime() on a stage authoring timeCodesPerSecond = 0
// segfaults inside OpenUSD -- UsdGeomXformCache::GetLocalToWorldTransform() ->
// SdfData::GetBracketingTimeSamplesForPath(). Reproducible in stock pxr with no
// Slic3r involved, and it is not a contrived shape: the USD Working Group's own
// conformance suite ships one, in
// test_assets/foundation/stage_configuration/timeCodesPerSecond/. Such a stage
// has no meaningful earliest sample anyway, so fall back rather than crash.
UsdTimeCode read_time(const UsdStageRefPtr &stage)
{
    const double per_second = stage->GetTimeCodesPerSecond();
    if (!std::isfinite(per_second) || per_second <= 0.0) {
        BOOST_LOG_TRIVIAL(warning)
            << "load_usd: stage declares timeCodesPerSecond = " << per_second
            << ", so time samples cannot be placed; reading at the default time.";
        return UsdTimeCode::Default();
    }
    return UsdTimeCode::EarliestTime();
}

// UsdGeomCube, Sphere, Cylinder and Cone are geometry in their own right, not
// meshes -- a stage may contain nothing else. Apple's ModelIO tessellates them,
// so skipping them left this importer behind the path it replaces: measured over
// 3142 published USD files, 351 held only intrinsic shapes and were refused
// outright, and another 59 lost the shapes from a stage that also had meshes.
// `def Sphere "sphere" {}` is the simplest valid USD file there is.
//
// Built with Slic3r's own primitive builders at the facet angle Orca uses for
// shapes added from its own menu -- PI/90, two degrees, per GUI_ObjectList.cpp
// -- so an imported sphere is exactly as round as one Orca makes itself.
//
// USD centres all four on the prim origin, while its_make_cube() starts at a
// corner and its_make_cylinder()/its_make_cone() start at z = 0, so each is
// re-centred. `axis` lays a cylinder or cone along X or Y instead of Z.
//
// Capsule and Plane are deliberately absent: a capsule needs composing from a
// cylinder and two hemispheres, and a plane is a zero-thickness sheet that
// encloses no volume. Both are counted and named rather than passed over.
bool tessellate_gprim(const UsdPrim &prim, const UsdTimeCode when, indexed_triangle_set &its)
{
    const double facet = PI / 90.0;

    auto recentre = [&its](float dx, float dy, float dz) {
        for (Vec3f &v : its.vertices) v += Vec3f(dx, dy, dz);
    };
    // Slic3r builds along +Z; rotate that axis onto X or Y as the prim asks.
    auto orient = [&its](const TfToken &axis) {
        if (axis == UsdGeomTokens->x)
            for (Vec3f &v : its.vertices) v = Vec3f(v.z(), v.y(), -v.x());
        else if (axis == UsdGeomTokens->y)
            for (Vec3f &v : its.vertices) v = Vec3f(v.x(), v.z(), -v.y());
    };

    if (UsdGeomCube cube = UsdGeomCube(prim)) {
        double size = 2.0;
        cube.GetSizeAttr().Get(&size, when);
        if (!(size > 0.0)) return false;
        its = its_make_cube(size, size, size);
        recentre(float(-0.5 * size), float(-0.5 * size), float(-0.5 * size));
        return true;
    }
    if (UsdGeomSphere sphere = UsdGeomSphere(prim)) {
        double radius = 1.0;
        sphere.GetRadiusAttr().Get(&radius, when);
        if (!(radius > 0.0)) return false;
        its = its_make_sphere(radius, facet);   // already centred on the origin
        return true;
    }
    if (UsdGeomCylinder cylinder = UsdGeomCylinder(prim)) {
        double radius = 1.0, height = 2.0;
        TfToken axis = UsdGeomTokens->z;
        cylinder.GetRadiusAttr().Get(&radius, when);
        cylinder.GetHeightAttr().Get(&height, when);
        cylinder.GetAxisAttr().Get(&axis, when);
        if (!(radius > 0.0) || !(height > 0.0)) return false;
        its = its_make_cylinder(radius, height, facet);
        recentre(0.f, 0.f, float(-0.5 * height));
        orient(axis);
        return true;
    }
    if (UsdGeomCone cone = UsdGeomCone(prim)) {
        double radius = 1.0, height = 2.0;
        TfToken axis = UsdGeomTokens->z;
        cone.GetRadiusAttr().Get(&radius, when);
        cone.GetHeightAttr().Get(&height, when);
        cone.GetAxisAttr().Get(&axis, when);
        if (!(radius > 0.0) || !(height > 0.0)) return false;
        its = its_make_cone(radius, height, facet);
        recentre(0.f, 0.f, float(-0.5 * height));
        orient(axis);
        return true;
    }
    return false;
}

// World map (mm scale + Y-up->Z-up) and USD->Eigen matrix conversion; defined below with the
// instancing helpers but needed earlier by read_stage to place a deferred cage.
Transform3d stage_root(double scale, bool y_up);
Transform3d gf_to_transform(const GfMatrix4d &m);

// One mesh per USD prim, so a stage's structure survives into Orca's model
// rather than being flattened on the way in.
struct NamedMesh
{
    std::string          name;
    indexed_triangle_set its;
    // Set on the amplify path: a cage deferred to slice time, with `its` its local control mesh and
    // `xform` its world placement. Absent -> `its` is finished world-space geometry (eager path).
    std::optional<PrintMan::SubdivCage> cage;
    Transform3d                         xform = Transform3d::Identity();
    // OSL shaders resolved from the prim's bound UsdShade material -- one per terminal (see PrintManScene).
    std::string                         osl_surface_shader;        // material:surface      -> Cout
    std::string                         osl_displacement_shader;   // material:displacement -> Disp
    double                              osl_max_displacement = 0.0;
    // Colour-shader hints, read as inputs on the surface shader; see PrintManScene.
    bool                                color_object_space  = false;
    bool                                color_dither        = false;
    bool                                color_all_filaments = false;
    // Authored shader parameters on material 0's surface/displacement shaders (see PrintManScene).
    std::vector<PrintMan::ShaderParam>  surface_params;
    std::vector<PrintMan::ShaderParam>  displacement_params;
    // Region materials beyond the prim's own (material 0, the scalar fields above): materials bound by
    // UsdGeomSubset face regions, selected per control face by `cage->face_material`. Empty = single material.
    std::vector<PrintMan::MaterialShaders> extra_materials;
};

// Why the counters exist: every reason a mesh is skipped has to reach the user.
// A stage where all forty meshes are `proxy` must not report only that no
// printable meshes were found,
// and one mesh dropped from a stage of ten must not import silently.
struct SkipCounts
{
    size_t purpose = 0;      // proxy or guide
    size_t invisible = 0;    // computed visibility is `invisible`
    size_t no_data = 0;      // points/counts/indices unreadable or empty
    size_t bad_topology = 0; // counts and indices disagree, or an index is out of range
    size_t holes = 0;        // authored holeIndices, which are not honoured
    size_t degenerate = 0;   // fewer than two triangles, so not a printable solid
    size_t nonfinite = 0;    // a point coordinate is NaN or infinite
    size_t gprim = 0;        // an intrinsic shape this does not tessellate
};

std::string describe(const SkipCounts &s)
{
    std::string out;
    auto add = [&out](size_t n, const char *what) {
        if (!n) return;
        if (!out.empty()) out += ", ";
        out += std::to_string(n) + " " + what;
    };
    add(s.purpose,      "skipped as proxy/guide");
    add(s.invisible,    "skipped as invisible");
    add(s.no_data,      "skipped with unreadable point data");
    add(s.bad_topology, "skipped with inconsistent topology");
    add(s.holes,        "skipped for authored holeIndices");
    add(s.degenerate,   "skipped as single triangles");
    add(s.nonfinite,    "skipped for non-finite coordinates");
    add(s.gprim,        "skipped as untessellated intrinsic shapes");
    return out.empty() ? out : "USD import: " + out + ".";
}

// `purpose` and `visibility` are INHERITED, so the local opinion is the wrong
// question. Real content authors them on a group -- `/World/Proxy` with a plain
// Mesh underneath -- and reading the leaf's own attribute returns the fallback
// `default`, importing exactly the stand-in geometry this is meant to exclude.
// ComputePurpose()/ComputeVisibility() resolve down the hierarchy.
bool is_renderable(const UsdPrim &prim, const UsdTimeCode when, SkipCounts &skipped)
{
    UsdGeomImageable imageable(prim);
    if (!imageable)
        return false;

    const TfToken purpose = imageable.ComputePurpose();
    if (purpose == UsdGeomTokens->proxy || purpose == UsdGeomTokens->guide) {
        ++ skipped.purpose;
        return false;
    }
    if (imageable.ComputeVisibility(when) == UsdGeomTokens->invisible) {
        ++ skipped.invisible;
        return false;
    }
    return true;
}

// Read primvars:st into one (u,v) per face-corner (parallel to face_indices), resolving the primvar's
// interpolation -- faceVarying is one value per corner (what a seamed sphere authors, and what the engine
// refines face-varyingly); vertex/varying is one per point, looked up by the corner's vertex. Any indexing
// is honoured. Empty when the prim carries no st, or st is constant/uniform (not a per-surface texture map).
std::vector<std::array<double, 2>> read_st_primvar(const UsdGeomMesh &mesh, UsdTimeCode when,
                                                   const std::vector<int> &face_indices)
{
    std::vector<std::array<double, 2>> out;
    const UsdGeomPrimvar pv = UsdGeomPrimvarsAPI(mesh.GetPrim()).GetPrimvar(TfToken("st"));
    VtVec2fArray vals;
    if (! pv || ! pv.HasValue() || ! pv.Get(&vals, when) || vals.empty())
        return out;
    VtIntArray idx;
    const bool     indexed = pv.IsIndexed() && pv.GetIndices(&idx, when) && ! idx.empty();
    const TfToken  interp  = pv.GetInterpolation();
    const size_t   C       = face_indices.size();
    // The array actually indexed into per lookup: the index array if indexed, else the value array.
    const size_t   primary = indexed ? idx.size() : vals.size();
    auto sample = [&](long long lookup) -> std::array<double, 2> {
        const long long j = indexed ? (lookup >= 0 && lookup < (long long) idx.size() ? idx[lookup] : -1) : lookup;
        if (j < 0 || j >= (long long) vals.size()) return {0.0, 0.0};
        return {double(vals[j][0]), double(vals[j][1])};
    };
    if (interp == UsdGeomTokens->faceVarying) {
        if (primary < C) {   // too few st for the face-corners -> refuse rather than zero-fill a partial UV
            BOOST_LOG_TRIVIAL(warning) << "load_usd: primvars:st (faceVarying) has " << primary
                << " entries for " << C << " face-corners; ignoring the UV.";
            return out;
        }
        out.resize(C);
        for (size_t i = 0; i < C; ++i) out[i] = sample((long long) i);              // per face-corner
    } else if (interp == UsdGeomTokens->vertex || interp == UsdGeomTokens->varying) {
        int max_v = -1;
        for (int v : face_indices) max_v = std::max(max_v, v);
        if (primary <= size_t(std::max(max_v, 0))) {   // too few st to cover every referenced point
            BOOST_LOG_TRIVIAL(warning) << "load_usd: primvars:st (vertex) has " << primary
                << " entries but a face references point " << max_v << "; ignoring the UV.";
            return out;
        }
        out.resize(C);
        for (size_t i = 0; i < C; ++i) out[i] = sample((long long) face_indices[i]); // per point, via the corner's vertex
    }
    return out;   // constant/uniform -> left empty
}

// Read a mesh's subdivision attributes (creases, corners, interpolateBoundary, triangleSubdivisionRule)
// into a local-frame SubdivCage carried to slice time. Winding stays the raw `left_handed` (mirror is
// held in the placement xform, flipped per placement). When `compute_aabb`, the cage is refined once at
// `level` = kDeviceLevelMax (the finest, most contracted slice) only to record the limit AABB for bed
// placement, then discarded; the eager path passes false. Shared by the standalone and instanced paths.
PrintMan::SubdivCage read_subdiv_cage(const UsdGeomMesh &mesh, UsdTimeCode when,
                                      const std::vector<usd_subdiv::Pt> &verts,
                                      const std::vector<int> &counts, const std::vector<int> &indices,
                                      const std::string &scheme, bool left_handed, int level,
                                      bool compute_aabb = true)
{
    VtIntArray   creaseI, creaseL, cornerI;
    VtFloatArray creaseS, cornerS;
    mesh.GetCreaseIndicesAttr().Get(&creaseI, when);
    mesh.GetCreaseLengthsAttr().Get(&creaseL, when);
    mesh.GetCreaseSharpnessesAttr().Get(&creaseS, when);
    mesh.GetCornerIndicesAttr().Get(&cornerI, when);
    mesh.GetCornerSharpnessesAttr().Get(&cornerS, when);
    TfToken bnd, tri;
    mesh.GetInterpolateBoundaryAttr().Get(&bnd, when);
    mesh.GetTriangleSubdivisionRuleAttr().Get(&tri, when);
    const std::string b = bnd.GetString();
    const int boundary = b == "none"     ? usd_subdiv::BOUNDARY_NONE
                       : b == "edgeOnly" ? usd_subdiv::BOUNDARY_EDGE_ONLY
                                         : usd_subdiv::BOUNDARY_EDGE_AND_CORNER;
    const bool tri_smooth = (tri.GetString() == "smooth");
    const std::vector<int>    ci(creaseI.begin(), creaseI.end());
    const std::vector<int>    cl(creaseL.begin(), creaseL.end());
    const std::vector<double> cs(creaseS.begin(), creaseS.end());
    const std::vector<int>    ki(cornerI.begin(), cornerI.end());
    const std::vector<double> ks(cornerS.begin(), cornerS.end());

    PrintMan::SubdivCage sc;
    sc.points.assign(verts.begin(), verts.end());
    sc.face_counts     = counts;
    sc.face_indices    = indices;
    sc.crease_indices  = ci;
    sc.crease_lengths  = cl;
    sc.crease_sharp    = cs;
    sc.corner_indices  = ki;
    sc.corner_sharp    = ks;
    sc.st              = read_st_primvar(mesh, when, indices);   // authored UV (face-varying) for texturing
    sc.boundary        = boundary;
    sc.triangle_smooth = tri_smooth;
    sc.flip_winding    = left_handed;
    sc.scheme          = scheme;

    if (compute_aabb) {
        // Limit AABB for bed placement; refined_limit_aabb converges-and-stops, so a large cage is not
        // refined to 4^level transient faces just to measure a box.
        const usd_subdiv::Cage cage = usd_subdiv::build_cage(verts, counts, indices, ci, cl, cs, ki, ks,
                                                             boundary, tri_smooth);
        usd_subdiv::refined_limit_aabb(cage, scheme, level, sc.refined_lo, sc.refined_hi);
    }
    return sc;
}

// The OSL shaders of a resolved UsdShade material, exactly as USD models shading: the material's `surface`
// and `displacement` terminal outputs each connect to a Shader prim whose info:id names a compiled OSL shader
// (.oso on the searchpath). One shader may back both terminals. Colour behaviour (objectSpace / dither /
// allFilaments) and the displacement clamp (maxMagnitude) are PrintMan slicer directives, not OSL parameters,
// so they are authored under the printman: input namespace (inputs:printman:*) to keep them from colliding
// with a real shader parameter of the same name.
// The shader's own authored inputs, forwarded to OSL so a material can tune its shader. The printman:
// namespace is host directives (maxMagnitude, objectSpace, ...) read separately, not shader parameters, so
// it is skipped here. v1 carries float and int (bool as int) -- the scalar types PrintMan shaders declare.
static std::vector<PrintMan::ShaderParam> read_shader_params(const UsdShadeShader &s, UsdTimeCode when)
{
    std::vector<PrintMan::ShaderParam> out;
    for (const UsdShadeInput &in : s.GetInputs()) {
        const std::string name = in.GetBaseName().GetString();
        if (name.rfind("printman:", 0) == 0)
            continue;   // a host directive, not a shader parameter
        const SdfValueTypeName t = in.GetTypeName();
        using T = PrintMan::ShaderParam::Type;
        PrintMan::ShaderParam p;
        p.name = name;
        // A 3-float value (colour/vector/point/normal), authored either single- or double-precision.
        auto push_vec3 = [&](T ty) {
            GfVec3f vf;
            if (in.Get(&vf, when)) { p.type = ty; p.x = vf[0]; p.y = vf[1]; p.z = vf[2]; out.push_back(p); return; }
            GfVec3d vd;
            if (in.Get(&vd, when)) { p.type = ty; p.x = vd[0]; p.y = vd[1]; p.z = vd[2]; out.push_back(p); }
        };
        if (t == SdfValueTypeNames->Float) {
            float v = 0.0f;
            if (in.Get(&v, when)) { p.type = T::Float; p.x = v; out.push_back(p); }
        } else if (t == SdfValueTypeNames->Double) {
            double v = 0.0;
            if (in.Get(&v, when)) { p.type = T::Float; p.x = v; out.push_back(p); }
        } else if (t == SdfValueTypeNames->Int) {
            int v = 0;
            if (in.Get(&v, when)) { p.type = T::Int; p.x = v; out.push_back(p); }
        } else if (t == SdfValueTypeNames->Bool) {
            bool v = false;
            if (in.Get(&v, when)) { p.type = T::Int; p.x = v ? 1.0 : 0.0; out.push_back(p); }
        } else if (t == SdfValueTypeNames->Color3f  || t == SdfValueTypeNames->Color3d) {
            push_vec3(T::Color);
        } else if (t == SdfValueTypeNames->Vector3f || t == SdfValueTypeNames->Vector3d
                   || t == SdfValueTypeNames->Float3 || t == SdfValueTypeNames->Double3) {
            push_vec3(T::Vector);
        } else if (t == SdfValueTypeNames->Point3f  || t == SdfValueTypeNames->Point3d) {
            push_vec3(T::Point);
        } else if (t == SdfValueTypeNames->Normal3f || t == SdfValueTypeNames->Normal3d) {
            push_vec3(T::Normal);
        } else if (t == SdfValueTypeNames->String) {
            std::string v;
            if (in.Get(&v, when)) { p.type = T::String; p.str = v; out.push_back(p); }
        } else if (t == SdfValueTypeNames->Token) {
            TfToken v;
            if (in.Get(&v, when)) { p.type = T::String; p.str = v.GetString(); out.push_back(p); }
        } else {
            // Loudly, not silently: an unsupported type is dropped, so the author knows why their parameter
            // did not take instead of the shader quietly keeping its default.
            BOOST_LOG_TRIVIAL(warning) << "PrintMan: shader parameter '" << name << "' has unsupported type '"
                << t.GetAsToken().GetString() << "'; not forwarded to OSL, so the shader keeps its .osl default.";
        }
    }
    return out;
}

static PrintMan::MaterialShaders read_material_shaders(const UsdShadeMaterial &mat, UsdTimeCode when)
{
    PrintMan::MaterialShaders pm;
    // surface terminal -> colour shader + its colour hints + its authored parameters
    if (const UsdShadeShader s = mat.ComputeSurfaceSource()) {
        TfToken id;
        if (s.GetShaderId(&id))
            pm.surface = id.GetString();
        if (const UsdShadeInput in = s.GetInput(TfToken("printman:objectSpace")))  in.Get(&pm.object_space,  when);
        if (const UsdShadeInput in = s.GetInput(TfToken("printman:dither")))       in.Get(&pm.dither,        when);
        if (const UsdShadeInput in = s.GetInput(TfToken("printman:allFilaments"))) in.Get(&pm.all_filaments, when);
        pm.surface_params = read_shader_params(s, when);
    }
    // displacement terminal -> relief shader + its clamp + its authored parameters
    if (const UsdShadeShader s = mat.ComputeDisplacementSource()) {
        TfToken id;
        if (s.GetShaderId(&id))
            pm.displacement = id.GetString();
        float m = 0.0f;
        if (const UsdShadeInput in = s.GetInput(TfToken("printman:maxMagnitude"))) { in.Get(&m, when); pm.max_displacement = m; }
        pm.displacement_params = read_shader_params(s, when);
    }
    return pm;
}

// The OSL shaders bound to a prim through its own material binding (UsdShade); empty/zero if the prim carries
// no binding. The mesh's whole-surface material; a face region's material is resolved from its GeomSubset.
PrintMan::MaterialShaders read_printman_material(const UsdPrim &prim, UsdTimeCode when)
{
    const UsdShadeMaterial mat = UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial();
    return mat ? read_material_shaders(mat, when) : PrintMan::MaterialShaders{};
}

// Region materials: a UsdGeomSubset of elementType "face" that binds its own material paints + displaces
// that face region in a different shader than the rest of the mesh. Fills `face_material` (per control face;
// 0 = the mesh's own material `mesh_mat`, k>=1 = extra_materials[k-1]) and appends the region materials to
// `extra_materials`. Materials are deduplicated by their bound Material prim path, and a subset re-binding the
// mesh's own material maps back to 0. Faces in no subset stay 0. No face subsets -> face_material stays empty,
// so the cage prints as a single material (byte-identical to the pre-subset path). nfaces is the control-face
// count; subset face indices are into that same authored faceVertexCounts order, which is exactly the core's
// per-control-face material tag (ctag). Only meaningful on the amplify path (a deferred cage).
static void read_face_subset_materials(const UsdGeomMesh &mesh, UsdTimeCode when, int nfaces,
                                       std::vector<int> &face_material,
                                       std::vector<PrintMan::MaterialShaders> &extra_materials)
{
    const std::vector<UsdGeomSubset> subsets = UsdGeomSubset::GetAllGeomSubsets(mesh);
    if (subsets.empty())
        return;
    // The mesh's own bound material prim maps to 0, so a subset re-binding it deduplicates to material 0.
    std::map<std::string, int> mat_index;
    if (const UsdShadeMaterial m0 = UsdShadeMaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial())
        mat_index[m0.GetPath().GetString()] = 0;
    std::vector<int> fm;   // allocated lazily -- stays empty (single material) until a subset overrides a face
    for (const UsdGeomSubset &subset : subsets) {
        TfToken elem;
        if (! subset.GetElementTypeAttr().Get(&elem, when) || elem != UsdGeomTokens->face)
            continue;   // only a face subset selects a material region
        const UsdShadeMaterial sm = UsdShadeMaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial();
        if (! sm)
            continue;   // a subset with no material binding repaints nothing
        // A subset that paints no in-range face (empty indices, or all out of range) is skipped whole,
        // before its material is recorded -- so it adds no phantom region material and cannot flip a cage
        // that should stay single-material onto the multi-material merge path.
        VtIntArray idx;
        subset.GetIndicesAttr().Get(&idx, when);
        bool paints = false;
        for (const int f : idx)
            if (f >= 0 && f < nfaces) { paints = true; break; }
        if (! paints)
            continue;
        const std::string key = sm.GetPath().GetString();
        int mi;
        if (const auto it = mat_index.find(key); it != mat_index.end())
            mi = it->second;
        else {
            mi = int(extra_materials.size()) + 1;
            extra_materials.push_back(read_material_shaders(sm, when));   // sm already resolved above
            mat_index[key] = mi;
        }
        if (mi == 0)
            continue;   // this subset re-binds the mesh's own material: no per-face override needed
        if (fm.empty())
            fm.assign(nfaces, 0);
        for (const int f : idx)
            if (f >= 0 && f < nfaces)
                fm[f] = mi;
    }
    if (! fm.empty())
        face_material = std::move(fm);
}

// amplify defers a subdivision cage to slice time (emits the control mesh + a SubdivCage); without it
// (the TriangleMesh merge path and the amplify=false reference) a cage is refined eagerly here.
bool read_stage(const char *path, std::vector<NamedMesh> &out, std::string &message, bool amplify = false)
{
    try {
        if (!register_usd_plugins()) {
            message = "OpenUSD's plugin data is missing from this build, so USD files cannot be read.";
            return false;
        }

        // TfErrorMark catches USD's non-fatal diagnostics so they can be
        // reported rather than only logged by USD itself. It cannot catch a
        // TF_FATAL_ERROR, which calls abort() -- see the note above about the
        // plugin registry, which is the reachable cause of that.
        TfErrorMark usd_errors;

        UsdStageRefPtr stage = UsdStage::Open(path);
        if (!stage) {
            message = "Could not open USD stage " + std::string(path) + ".";
            for (const auto &e : usd_errors)
                message += " " + e.GetCommentary();
            BOOST_LOG_TRIVIAL(error) << "load_usd: " << message;
            return false;
        }

        const double scale = mm_per_unit(stage);
        const bool   y_up  = (UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->y);
        const UsdTimeCode when = read_time(stage);

        UsdGeomXformCache    xf_cache(when);
        size_t               mesh_count = 0;
        size_t               cages      = 0;   // imported unsubdivided, not skipped
        SkipCounts           skipped;

        // Instance proxies: a plain Traverse() returns instanceable prims but not
        // their prototype children, so an instanced scene yields zero meshes and
        // is refused as empty -- a file that plainly contains geometry.
        //
        // The predicate is USD's default one, which is
        // IsActive && IsDefined && IsLoaded && !IsAbstract -- spelling it out by
        // hand and omitting IsActive would import prims deactivated in an
        // override layer, and `active = false` is how a USD pipeline deletes
        // geometry non-destructively. !IsAbstract keeps class prototypes from
        // being counted as geometry alongside the instances referencing them.
        UsdPrimRange range = UsdPrimRange::Stage(stage, UsdTraverseInstanceProxies());

        for (const UsdPrim &prim : range) {
            UsdGeomMesh mesh(prim);
            if (!mesh)
                continue;
            if (!is_renderable(prim, when, skipped))
                continue;

            VtVec3fArray points;
            VtIntArray   counts, indices;
            if (!mesh.GetPointsAttr().Get(&points, when) ||
                !mesh.GetFaceVertexCountsAttr().Get(&counts, when) ||
                !mesh.GetFaceVertexIndicesAttr().Get(&indices, when) ||
                points.empty() || counts.empty()) {
                // Never a bare `continue`. A mesh dropped here used to vanish
                // from a mixed stage with nothing said and the import still
                // reporting success, which is the failure this file exists to
                // prevent.
                ++ skipped.no_data;
                continue;
            }

            // Absent subdivisionScheme means catmullClark (USD default; mayaUSD omits the default),
            // reported distinctly so an author who meant a polygon mesh knows to set it to "none".
            TfToken scheme = UsdGeomTokens->catmullClark;
            const bool declared = mesh.GetSubdivisionSchemeAttr().HasAuthoredValue();
            mesh.GetSubdivisionSchemeAttr().Get(&scheme, when);
            const bool is_cage = (scheme != UsdGeomTokens->none);

            // holeIndices deletes faces from the authored surface. Ignoring it
            // yields a closed solid where the author specified an opening, so it
            // is refused rather than silently filled.
            VtIntArray holes;
            if (mesh.GetHoleIndicesAttr().Get(&holes, when) && !holes.empty()) {
                BOOST_LOG_TRIVIAL(error)
                    << "load_usd: " << prim.GetPath().GetString() << " authors "
                    << holes.size() << " holeIndices, which are not honoured;"
                       " importing it would fill openings the author specified.";
                ++ skipped.holes;
                continue;
            }

            // The counts must describe exactly the indices present. A truncated
            // faceVertexIndices is a common broken-export shape, and dropping the
            // affected faces one at a time produces a mesh with a hole in it and
            // a successful-looking import.
            // Summed over every face, including a degenerate one: stopping at the
            // first left a partial total that was then reported as what the counts
            // describe, so [4, 2, 4] against ten indices said "describe 4" when
            // they describe 10. The two faults are also reported separately --
            // quoting a length mismatch of "10 indices but 10" to say a face was
            // malformed explained nothing.
            size_t needed = 0;
            bool   degenerate = false;
            for (int n : counts) {
                if (n < 3) degenerate = true;
                if (n > 0) needed += size_t(n);
            }
            if (degenerate || needed != indices.size()) {
                std::string why;
                if (needed != indices.size())
                    why = "faceVertexCounts describe " + std::to_string(needed) +
                          " indices but faceVertexIndices holds " +
                          std::to_string(indices.size());
                if (degenerate)
                    why += (why.empty() ? "" : ", and ") +
                           std::string("a face has fewer than three vertices");
                BOOST_LOG_TRIVIAL(error)
                    << "load_usd: " << prim.GetPath().GetString()
                    << " has inconsistent topology: " << why << ".";
                ++ skipped.bad_topology;
                continue;
            }

            // USD permits inf and nan in a point array, and OpenUSD reads them
            // back faithfully. Nothing downstream does: the bounding box, the
            // volume and every slicing decision become nan, and the object
            // reports a successful import with no size. Apple's ModelIO refuses
            // such a file outright, so accepting it would be strictly worse than
            // the path this replaces. OpenUSD's own test suite ships one
            // (testUsdviewInfGeom/infGeom.usda), a pentagon with one vertex at
            // infinity.
            bool finite = true;
            for (const GfVec3f &p : points)
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
                    finite = false;
                    break;
                }
            if (!finite) {
                BOOST_LOG_TRIVIAL(error)
                    << "load_usd: " << prim.GetPath().GetString()
                    << " has a non-finite point coordinate (nan or inf), which"
                       " would give the imported object no measurable size.";
                ++ skipped.nonfinite;
                continue;
            }

            // Every index value must address a point of THIS mesh. Checking only
            // the cursor against the array length leaves the values unchecked,
            // and they land in an indexed_triangle_set whose consumers index
            // std::vector with operator[] -- an out-of-bounds read on the first
            // volume computation. A negative value is worse than a crash: it
            // silently addresses a previous mesh's vertices.
            const long long npoints = (long long) points.size();
            bool indices_ok = true;
            for (int v : indices)
                if (v < 0 || (long long) v >= npoints) { indices_ok = false; break; }
            if (!indices_ok) {
                BOOST_LOG_TRIVIAL(error)
                    << "load_usd: " << prim.GetPath().GetString()
                    << " has a faceVertexIndices value outside [0, " << npoints << ").";
                ++ skipped.bad_topology;
                continue;
            }

            // Each prim becomes its own mesh, so indices start at zero and the
            // int that Slic3r's triangle indices use only has to hold this
            // prim's vertex count. Refuse rather than wrap into garbage.
            if (points.size() > size_t(std::numeric_limits<int>::max())) {
                message = "Refusing " + std::string(path) + ": " +
                          prim.GetPath().GetString() + " has " +
                          std::to_string(points.size()) + " vertices, more than a"
                          " triangle index can address.";
                BOOST_LOG_TRIVIAL(error) << "load_usd: " << message;
                return false;
            }

            // leftHanded reverses the winding, and Slic3r derives facet
            // orientation from winding, so this must be honoured or the mesh
            // imports inside-out.
            TfToken orientation = UsdGeomTokens->rightHanded;
            mesh.GetOrientationAttr().Get(&orientation, when);
            const bool left_handed = (orientation == UsdGeomTokens->leftHanded);
            bool flip = left_handed;

            const GfMatrix4d world = xf_cache.GetLocalToWorldTransform(prim);
            // A mirror flips winding again; two flips cancel. The eager path bakes it in; a deferred
            // cage keeps the mirror in its placement xform, so its own winding stays raw left_handed.
            if (world.GetDeterminant() < 0.0)
                flip = !flip;

            // A cage is refined to its limit surface. Subdivision is affine-covariant, so refining the
            // control mesh in its local frame gives the same surface; the amplify path defers this to
            // slice time (level then follows the print resolution, viewport shows the cage), else here.
            std::vector<usd_subdiv::Pt> verts;
            verts.reserve(points.size());
            for (const GfVec3f &p : points) verts.push_back({p[0], p[1], p[2]});
            std::vector<int> face_counts(counts.begin(), counts.end());
            std::vector<int> face_indices(indices.begin(), indices.end());

            std::optional<PrintMan::SubdivCage> deferred_cage;
            Transform3d                         cage_xform = Transform3d::Identity();

            if (is_cage) {
                if (amplify) {
                    // Defer: `verts` stays the control mesh (proxy), the SubdivCage carries the surface.
                    // Its AABB is at kDeviceLevelMax (finest slice) so the object never floats off the bed.
                    deferred_cage = read_subdiv_cage(mesh, when, verts, face_counts, face_indices,
                                                     scheme.GetString(), left_handed, usd_subdiv::kDeviceLevelMax);
                    cage_xform    = stage_root(scale, y_up) * gf_to_transform(world);
                } else {
                    // Eager (amplify=false reference): refine in place at the fixed level; verts/counts/
                    // indices become the refined surface, baked to world below like any other mesh.
                    const PrintMan::SubdivCage sc = read_subdiv_cage(
                        mesh, when, verts, face_counts, face_indices, scheme.GetString(), left_handed,
                        usd_subdiv::kDeviceLevelMin, /*compute_aabb=*/false);
                    usd_subdiv::Cage cage = usd_subdiv::subdivide(
                        usd_subdiv::build_cage(verts, face_counts, face_indices,
                                               sc.crease_indices, sc.crease_lengths, sc.crease_sharp,
                                               sc.corner_indices, sc.corner_sharp, sc.boundary,
                                               sc.triangle_smooth),
                        scheme.GetString(), usd_subdiv::kDeviceLevelMin);
                    verts = cage.verts;
                    face_counts.clear();
                    for (int f = 0; f < cage.nfaces(); ++ f)
                        face_counts.push_back(cage.foff[f + 1] - cage.foff[f]);
                    face_indices = cage.fvi;
                }
            }

            // A deferred cage stays in LOCAL space (its placement xform applies the world transform at
            // slice time); everything else bakes the world / up-axis / scale transform here.
            indexed_triangle_set its;
            its.vertices.reserve(verts.size());
            for (const usd_subdiv::Pt &p : verts) {
                double x, y, z;
                if (deferred_cage) {
                    x = p[0]; y = p[1]; z = p[2];
                } else {
                    GfVec3d w = world.Transform(GfVec3d(p[0], p[1], p[2]));
                    // Y-up to Z-up is a +90 deg rotation about X: (x, y, z) -> (x, -z, y).
                    // The sign matters and is not symmetric -- the inverse rotation
                    // has identical extents but places the model below the plate.
                    x = w[0]; y = w[1]; z = w[2];
                    if (y_up) { double t = y; y = -z; z = t; }
                    x *= scale; y *= scale; z *= scale;
                }
                its.vertices.emplace_back(float(x), float(y), float(z));
            }

            // Fan-triangulate each n-gon. Authored content is mostly quads and subdivision
            // emits quads or triangles; a fan is correct for any convex face, and concave
            // faces produce overlapping triangles the slicer's boolean resolves anyway. A deferred
            // cage carries only left_handed here (the coarse proxy is local); its mirror is applied
            // at slice time.
            const bool tri_flip = deferred_cage ? left_handed : flip;
            size_t cursor = 0;
            for (int n : face_counts) {
                for (int k = 1; k + 1 < n; ++ k) {
                    int a = face_indices[cursor];
                    int b = face_indices[cursor + k];
                    int c = face_indices[cursor + k + 1];
                    if (tri_flip) std::swap(b, c);
                    its.indices.emplace_back(a, b, c);
                }
                cursor += size_t(n);
            }
            // A ModelVolume built from one triangle never gets a convex hull:
            // Model.hpp guards calculate_convex_hull() behind facets_count() > 1,
            // and ModelVolume::calculate_convex_hull_2d() then dereferences the
            // null m_convex_hull, so the first hull computation segfaults. One
            // ModelVolume per prim makes that reachable from an ordinary file --
            // Apple's toy_drummer.usdz has seven single-triangle prims left over
            // from its rig, and crashed the slicer on import. A lone triangle
            // encloses no volume and cannot be printed, so it is dropped, and
            // counted rather than dropped quietly.
            if (its.indices.size() < 2) {
                BOOST_LOG_TRIVIAL(warning)
                    << "load_usd: " << prim.GetPath().GetString() << " has only "
                    << its.indices.size() << " triangle(s), which encloses no"
                       " volume; skipping it.";
                ++ skipped.degenerate;
                continue;
            }

            // OSL shaders bound to the prim through its UsdShade material (surface -> Cout, displacement ->
            // Disp), with colour hints and the displacement clamp read as inputs on those shaders. Carried to
            // the scene and applied at slice time when the build links OSL (ignored otherwise). Only meaningful
            // for a deferred cage (the amplify path).
            PrintMan::MaterialShaders               pm;
            std::vector<PrintMan::MaterialShaders>  extra_materials;
            if (deferred_cage) {
                pm = read_printman_material(mesh.GetPrim(), when);
                // A UsdGeomSubset may bind a different material to a face region -- per-control-face material
                // tags (material 0 = pm, k>=1 = extra_materials[k-1]) on the cage, so each region slices in
                // its own shader. Empty face_material -> single material, the pre-subset path unchanged.
                read_face_subset_materials(mesh, when, int(deferred_cage->face_counts.size()),
                                           deferred_cage->face_material, extra_materials);
            }

            out.push_back({prim.GetPath().GetString(), std::move(its), std::move(deferred_cage),
                           cage_xform, pm.surface, pm.displacement, pm.max_displacement,
                           pm.object_space, pm.dither, pm.all_filaments,
                           std::move(pm.surface_params), std::move(pm.displacement_params),
                           std::move(extra_materials)});
            ++ mesh_count;

            // Counted only once the mesh is actually emitted. Counting it at the
            // scheme check instead let a cage that was then skipped for
            // holeIndices or bad topology report "1 of the 0 mesh(es) imported
            // are subdivision cages", and claim in the log that a control mesh
            // had been imported when nothing had been.
            if (is_cage) {
                if (cages < 5)
                    BOOST_LOG_TRIVIAL(info)
                        << "load_usd: " << prim.GetPath().GetString() << " is a "
                        << scheme.GetString() << " subdivision cage"
                        << (declared ? "" : " (subdivisionScheme is absent, which USD"
                                            " defines as catmullClark)")
                        << (amplify ? "; shown as its control cage and refined to its subdivision"
                                      " surface at slice time."
                                    : "; refined to its subdivision surface.");
                ++ cages;
            }
        }

        // Second pass: geometry that is not a Mesh. Kept separate from the loop
        // above because none of that loop's validation applies -- these shapes
        // are generated, not read, so their topology is correct by construction.
        for (const UsdPrim &prim : UsdPrimRange::Stage(stage, UsdTraverseInstanceProxies())) {
            if (prim.IsA<UsdGeomMesh>() || !prim.IsA<UsdGeomGprim>())
                continue;
            if (!is_renderable(prim, when, skipped))
                continue;

            indexed_triangle_set its;
            if (!tessellate_gprim(prim, when, its)) {
                // Named, not merely counted: "a Capsule was skipped" is
                // actionable, "something was skipped" is not.
                BOOST_LOG_TRIVIAL(error)
                    << "load_usd: " << prim.GetPath().GetString() << " is a "
                    << prim.GetTypeName().GetString()
                    << ", which this importer does not tessellate; skipping it.";
                ++ skipped.gprim;
                continue;
            }

            TfToken orientation = UsdGeomTokens->rightHanded;
            UsdGeomGprim(prim).GetOrientationAttr().Get(&orientation, when);
            bool flip = (orientation == UsdGeomTokens->leftHanded);

            const GfMatrix4d world = xf_cache.GetLocalToWorldTransform(prim);
            if (world.GetDeterminant() < 0.0)
                flip = !flip;

            for (Vec3f &v : its.vertices) {
                GfVec3d w = world.Transform(GfVec3d(v.x(), v.y(), v.z()));
                double x = w[0], y = w[1], z = w[2];
                if (y_up) { double t = y; y = -z; z = t; }
                v = Vec3f(float(x * scale), float(y * scale), float(z * scale));
            }
            if (flip)
                for (stl_triangle_vertex_indices &t : its.indices)
                    std::swap(t[1], t[2]);

            out.push_back({prim.GetPath().GetString(), std::move(its)});
            ++ mesh_count;
        }

        // Every skip is stated even on success -- nine of ten meshes must not look like a clean import.
        // Cages are counted too. Model::read_from_file reads `message` only on failure, so this reaches
        // the log regardless.
        // Refusals do reach the error dialog. See the note in USD.hpp.
        // The skip tally rides along on every refusal below. The cage note does
        // NOT: it describes meshes that were imported, so attaching it to a
        // refusal asserts an import that did not happen. A stage of one
        // importable cage and one holed mesh otherwise put "1 of the 1 mesh(es)
        // imported are subdivision cages ... re-export to silence this" into the
        // error dialog for a load that imported nothing and silenced nothing.
        const std::string tally = describe(skipped);
        const std::string cage_note = cages == 0 ? std::string() :
            std::to_string(cages) + " of the " + std::to_string(mesh_count) +
            " mesh(es) imported are subdivision cages, refined to their subdivision"
            " surface (creases, corners and boundaries honoured)" +
            (amplify ? std::string(" at slice time, at a fixed level;")
                     : std::string(" at a fixed level;")) +
            " a high-curvature cage may still print slightly under-refined.";

        // holeIndices used to be counted as bad_topology and so was covered by
        // that refusal; with its own counter it needs its own, or a stage of
        // nothing but holed meshes falls through to the generic no-printable-
        // meshes error.
        if (skipped.holes > 0) {
            message = "Refusing " + std::string(path) + ": " +
                      std::to_string(skipped.holes) +
                      " mesh(es) author holeIndices, which are not honoured;"
                      " importing them would fill openings the author specified." +
                      (tally.empty() ? std::string() : " " + tally);
            BOOST_LOG_TRIVIAL(error) << "load_usd: " << message;
            return false;
        }
        if (skipped.bad_topology > 0) {
            message = "Refusing " + std::string(path) + ": " +
                      std::to_string(skipped.bad_topology) +
                      " mesh(es) have inconsistent topology (see the log for each)." +
                      (tally.empty() ? std::string() : " " + tally);
            BOOST_LOG_TRIVIAL(error) << "load_usd: " << message;
            return false;
        }
        if (out.empty()) {
            message = "No printable meshes in " + std::string(path) +
                      (tally.empty() ? "." : ". " + tally);
            BOOST_LOG_TRIVIAL(error) << "load_usd: " << message;
            return false;
        }
        // Success: both parts, and logged once. The cage note used to be logged
        // here and again where it was built.
        message = tally;
        if (!cage_note.empty())
            message = message.empty() ? "USD import: " + cage_note
                                      : message + " " + cage_note;
        if (!message.empty())
            BOOST_LOG_TRIVIAL(warning) << "load_usd: " << message;
        size_t triangles = 0;
        for (const NamedMesh &m : out)
            triangles += m.its.indices.size();
        BOOST_LOG_TRIVIAL(info)
            << "load_usd: " << mesh_count << " mesh(es), " << triangles
            << " triangles, " << scale << " mm/unit, " << (y_up ? "Y-up" : "Z-up");
    } catch (const std::exception &e) {
        // Without this the caller raises the generic "Loading of a model file
        // failed", which is the one failure that does not say why -- and OpenUSD
        // throws here for a corrupt .usdz or an allocation failure on a large
        // points array.
        message = "Could not read " + std::string(path) + ": " + e.what();
        BOOST_LOG_TRIVIAL(error) << "load_usd: " << message;
        return false;
    }
    return true;
}

// --- PrintMan instancing: build a scene, not a flattened pile of copies ---------------

// Convert a USD GfMatrix4d (row-vector) to a column-vector Eigen Transform3d: the plain
// transpose, since USD keeps translation in row 3 and Eigen in column 3.
Transform3d gf_to_transform(const GfMatrix4d &m)
{
    Transform3d t;
    for (int i = 0; i < 4; ++ i)
        for (int j = 0; j < 4; ++ j)
            t.matrix()(i, j) = m[j][i];
    return t;
}

// The stage's world map: uniform mm scale plus the Y-up->Z-up swap (x,y,z)->(x,-z,y) when
// the stage is Y-up. Folded into each placement, so prototypes stay in their local frame.
Transform3d stage_root(double scale, bool y_up)
{
    Transform3d root = Transform3d::Identity();
    if (y_up) {
        Eigen::Matrix3d swap;
        swap << 1, 0, 0,
                0, 0, -1,
                0, 1, 0;
        root.linear() = swap;
    }
    root.linear() *= scale;
    return root;
}

// Read a mesh prim's points as authored (prototype-local), fan-triangulated. leftHanded winding
// is honoured; a mirror (det<0) is left to the placement, flipped per-placement in slice_scene.
// False (with a tally) when the prim has no usable printable geometry.
bool read_prototype_mesh(const UsdGeomMesh &mesh, UsdTimeCode when,
                         SkipCounts &skipped, indexed_triangle_set &out)
{
    VtVec3fArray points;
    VtIntArray   counts, indices;
    if (!mesh.GetPointsAttr().Get(&points, when) ||
        !mesh.GetFaceVertexCountsAttr().Get(&counts, when) ||
        !mesh.GetFaceVertexIndicesAttr().Get(&indices, when) ||
        points.empty() || counts.empty()) {
        ++ skipped.no_data;
        return false;
    }
    // The same refusals read_stage makes for a standalone mesh, so an instanced prototype cannot slip
    // geometry past them. holeIndices delete faces: honouring them is unimplemented, and ignoring them
    // fills an opening the author specified. A non-finite point (NaN/inf) gives the instance no
    // measurable size and, through its proxy box, poisons the whole scene's bounding box.
    VtIntArray holes;
    if (mesh.GetHoleIndicesAttr().Get(&holes, when) && !holes.empty()) {
        BOOST_LOG_TRIVIAL(error)
            << "load_usd: " << mesh.GetPrim().GetPath().GetString() << " authors "
            << holes.size() << " holeIndices, which are not honoured;"
               " importing it would fill openings the author specified.";
        ++ skipped.holes;
        return false;
    }
    for (const GfVec3f &p : points)
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
            BOOST_LOG_TRIVIAL(error)
                << "load_usd: " << mesh.GetPrim().GetPath().GetString()
                << " has a non-finite point coordinate (nan or inf), which"
                   " would give the imported object no measurable size.";
            ++ skipped.nonfinite;
            return false;
        }

    TfToken orientation = UsdGeomTokens->rightHanded;
    mesh.GetOrientationAttr().Get(&orientation, when);
    const bool flip = (orientation == UsdGeomTokens->leftHanded);

    out.vertices.clear();
    out.vertices.reserve(points.size());
    for (const GfVec3f &p : points)
        out.vertices.emplace_back(float(p[0]), float(p[1]), float(p[2]));   // authored local

    out.indices.clear();
    size_t cursor = 0;
    for (int n : counts) {
        if (n < 3 || cursor + size_t(n) > indices.size()) {
            ++ skipped.bad_topology;
            return false;
        }
        for (int k = 1; k + 1 < n; ++ k) {
            int a = indices[cursor], b = indices[cursor + k], c = indices[cursor + k + 1];
            if (flip) std::swap(b, c);
            if (a < 0 || b < 0 || c < 0 ||
                size_t(a) >= points.size() || size_t(b) >= points.size() ||
                size_t(c) >= points.size()) {
                ++ skipped.bad_topology;
                return false;
            }
            out.indices.emplace_back(a, b, c);
        }
        cursor += size_t(n);
    }
    if (out.indices.size() < 2) {
        ++ skipped.degenerate;
        return false;
    }
    return true;
}

// If a prototype prim is a subdivision cage (scheme != none), read it into a SubdivCage so the slicer
// refines it per placement (the instancing analogue of the standalone deferred cage). False if not.
bool read_prototype_cage(const UsdGeomMesh &mesh, UsdTimeCode when, int level, PrintMan::SubdivCage &out)
{
    TfToken scheme = UsdGeomTokens->catmullClark;   // absent scheme means catmullClark, per the USD spec
    mesh.GetSubdivisionSchemeAttr().Get(&scheme, when);
    if (scheme == UsdGeomTokens->none)
        return false;

    VtVec3fArray points;
    VtIntArray   counts, indices;
    if (!mesh.GetPointsAttr().Get(&points, when) ||
        !mesh.GetFaceVertexCountsAttr().Get(&counts, when) ||
        !mesh.GetFaceVertexIndicesAttr().Get(&indices, when) ||
        points.empty() || counts.empty())
        return false;

    TfToken orientation = UsdGeomTokens->rightHanded;
    mesh.GetOrientationAttr().Get(&orientation, when);
    const bool left_handed = (orientation == UsdGeomTokens->leftHanded);

    std::vector<usd_subdiv::Pt> verts;
    verts.reserve(points.size());
    for (const GfVec3f &p : points) verts.push_back({p[0], p[1], p[2]});
    const std::vector<int> face_counts(counts.begin(), counts.end());
    const std::vector<int> face_indices(indices.begin(), indices.end());

    out = read_subdiv_cage(mesh, when, verts, face_counts, face_indices,
                           scheme.GetString(), left_handed, level);
    return true;
}

// Build a PrintManScene from a stage's instanceable prims: dedup prototypes by prim-in-prototype
// path, read each once in local space, one placement per instance proxy
// (root * transpose(local_to_world)). Left empty when the stage carries no instancing.
void build_instance_scene(const UsdStageRefPtr &stage, UsdTimeCode when,
                          const Transform3d &root, SkipCounts &skipped,
                          PrintMan::PrintManScene &scene)
{
    std::map<std::string, int> proto_index;
    UsdGeomXformCache xf(when);
    bool shading_set = false;   // one material set per scene: the first prototype to bind a material wins
    UsdPrimRange range = UsdPrimRange::Stage(stage, UsdTraverseInstanceProxies());
    for (const UsdPrim &prim : range) {
        if (!prim.IsInstanceProxy())
            continue;
        UsdGeomMesh mesh(prim);
        if (!mesh || !is_renderable(prim, when, skipped))
            continue;
        const std::string key = prim.GetPrimInPrototype().GetPath().GetString();
        int idx;
        auto found = proto_index.find(key);
        if (found == proto_index.end()) {
            indexed_triangle_set proto;
            if (!read_prototype_mesh(mesh, when, skipped, proto))
                continue;
            idx = int(scene.prototypes.size());
            scene.prototypes.push_back(std::move(proto));
            proto_index.emplace(key, idx);
            // A cage prototype is deferred like a standalone cage, refined per placement rather than
            // imported unrefined; the pushed prototype stays the control mesh (the arrangement proxy).
            PrintMan::SubdivCage sc;
            const bool is_cage = read_prototype_cage(mesh, when, usd_subdiv::kDeviceLevelMax, sc);
            // Carry the prototype's material so an instanced shaded stage prints shaded, not bare. The
            // scene's shader fields are ONE set applied to every cage placement; the same displacement field
            // is evaluated per placement in WORLD space (to_world_core_cage bakes each instance's transform),
            // so a non-uniform field -- e.g. wind on a lawn of one blade prototype -- bends each instance
            // to its own world position. Limitation (one material set per scene): the first prototype to bind
            // a material wins, and EVERY cage placement is then displaced/coloured by that one material --
            // including a second cage prototype that has no material of its own. Distinct per-prototype
            // shading needs per-prototype carriage (see PRINTMAN-RESUME 'shaders on instances'); until then a
            // stage of more than one cage prototype is warned below, not silently mixed.
            // The prototype's whole-surface material only; per-region face subsets on an instanced prototype
            // are a separate follow-up (single-material per prototype covers the grass-in-wind case).
            const PrintMan::MaterialShaders pm = read_printman_material(mesh.GetPrim(), when);
            if ((! pm.surface.empty() || ! pm.displacement.empty()) && ! shading_set) {
                scene.osl_surface_shader      = pm.surface;
                scene.osl_displacement_shader = pm.displacement;
                scene.osl_max_displacement    = pm.max_displacement;
                scene.color_object_space      = pm.object_space;
                scene.color_dither            = pm.dither;
                scene.color_all_filaments     = pm.all_filaments;
                scene.osl_surface_params      = pm.surface_params;
                scene.osl_displacement_params = pm.displacement_params;
                shading_set = true;
            }
            if (is_cage)
                scene.cages.emplace(idx, std::move(sc));
        } else {
            idx = found->second;
        }
        PrintMan::Placement place;
        place.prototype = idx;
        place.xform     = root * gf_to_transform(xf.GetLocalToWorldTransform(prim));
        scene.placements.push_back(place);
    }
    // One material set per scene: if a material was carried, it is applied to every cage placement -- so more
    // than one cage prototype means the others inherit it (their own material, if any, is dropped). Warn once;
    // this is the documented ceiling until per-prototype shader carriage lands.
    if (shading_set && scene.cages.size() > 1)
        BOOST_LOG_TRIVIAL(warning) << "PrintMan: instanced stage has " << scene.cages.size()
            << " subdivision-cage prototypes but one material set; every cage placement slices with the first "
               "shaded prototype's material and displacement (one material set per scene).";
}

// Reopen the stage (read_stage already validated it) to extract instancing as a scene.
// Returns false only on an open/plugin failure.
bool build_scene_from_path(const char *path, PrintMan::PrintManScene &scene)
{
    if (!register_usd_plugins())
        return false;
    UsdStageRefPtr stage = UsdStage::Open(path);
    if (!stage)
        return false;
    const Transform3d root = stage_root(mm_per_unit(stage),
                                        UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->y);
    SkipCounts skipped;
    build_instance_scene(stage, read_time(stage), root, skipped, scene);
    return true;
}

// A solid axis-aligned box spanning [lo, hi] (8 verts, 12 triangles).
indexed_triangle_set box_its(const Vec3f &lo, const Vec3f &hi)
{
    indexed_triangle_set b;
    b.vertices = {
        Vec3f(lo.x(), lo.y(), lo.z()), Vec3f(hi.x(), lo.y(), lo.z()),
        Vec3f(hi.x(), hi.y(), lo.z()), Vec3f(lo.x(), hi.y(), lo.z()),
        Vec3f(lo.x(), lo.y(), hi.z()), Vec3f(hi.x(), lo.y(), hi.z()),
        Vec3f(hi.x(), hi.y(), hi.z()), Vec3f(lo.x(), hi.y(), hi.z()),
    };
    const int f[12][3] = {
        {0, 3, 2}, {0, 2, 1}, {4, 5, 6}, {4, 6, 7},
        {0, 1, 5}, {0, 5, 4}, {1, 2, 6}, {1, 6, 5},
        {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7},
    };
    for (const auto &t : f)
        b.indices.emplace_back(t[0], t[1], t[2]);
    return b;
}

// One proxy box per placement, merged into one mesh, for the viewport / arrange / bed. For a mesh it is
// the EXACT vertex AABB (a box-corner over-estimate dips below a tilted instance and floats it off the
// bed). A cage uses its refined-surface AABB, not the wider control cage which would float the object.
indexed_triangle_set scene_proxy_boxes(const PrintMan::PrintManScene &scene)
{
    const double inf = std::numeric_limits<double>::infinity();
    indexed_triangle_set merged;
    for (const PrintMan::Placement &pl : scene.placements) {
        if (pl.prototype < 0 || size_t(pl.prototype) >= scene.prototypes.size()) continue;
        Vec3d lo = Vec3d::Constant(inf), hi = Vec3d::Constant(-inf);
        const auto cit = scene.cages.find(pl.prototype);
        if (cit != scene.cages.end()) {
            const std::array<double, 3> &rlo = cit->second.refined_lo;
            const std::array<double, 3> &rhi = cit->second.refined_hi;
            for (int c = 0; c < 8; ++ c) {
                const Vec3d corner((c & 1) ? rhi[0] : rlo[0],
                                   (c & 2) ? rhi[1] : rlo[1],
                                   (c & 4) ? rhi[2] : rlo[2]);
                const Vec3d w = pl.xform * corner;
                lo = lo.cwiseMin(w); hi = hi.cwiseMax(w);
            }
        } else {
            const indexed_triangle_set &proto = scene.prototypes[pl.prototype];
            if (proto.vertices.empty()) continue;
            for (const stl_vertex &v : proto.vertices) {
                const Vec3d w = pl.xform * v.cast<double>();
                lo = lo.cwiseMin(w); hi = hi.cwiseMax(w);
            }
        }
        // Grow the Z envelope up by the shader's max outward displacement so raised top caps get sliced;
        // leave the bottom on the AABB (downward caps clip against the bed, and dropping lo.z would sit
        // the object on sub-width rib tips -> empty first layer).
        if (const double md = scene.osl_max_displacement; md > 0.0)
            hi.z() += md;
        const indexed_triangle_set b = box_its(lo.cast<float>(), hi.cast<float>());
        const int base = int(merged.vertices.size());
        merged.vertices.insert(merged.vertices.end(), b.vertices.begin(), b.vertices.end());
        for (const stl_triangle_vertex_indices &t : b.indices)
            merged.indices.emplace_back(t[0] + base, t[1] + base, t[2] + base);
    }
    return merged;
}

// Attach a PrintMan scene to a new volume on `object`: the volume carries the proxy-box envelope
// (what arrange and ensure_on_bed read), and the scene is re-expressed in the centred frame
// add_volume introduces (-mesh_offset), so the +off in get_matrix() cancels instead of double-
// shifting params.trafo * get_matrix(). Shared by the instancing and standalone-cage paths.
// A self-contained .usdz bundles this model's OSL shaders (.oso) and texture maps (.tx) beside the
// geometry, so one file is the whole thing. USD reads the geometry from inside the package, but OSL and
// OIIO resolve shaders/textures on a *filesystem* searchpath -- so the bundled assets are extracted once
// to a cache dir keyed by the package (name + size + mtime, so re-authoring the same path re-extracts),
// reused across slices. Returns that dir, or empty when `path` is not a usdz or bundles no such assets
// (then the built-in PRINTMAN_OSL_SHADER_DIR is used). usdz stores its files uncompressed, so each entry's
// bytes are written straight out.
static std::string extract_usdz_shader_assets(const char *path)
{
    namespace fs = boost::filesystem;
    if (path == nullptr)
        return {};
    const std::string p = path;
    if (p.size() < 5 || p.compare(p.size() - 5, 5, ".usdz") != 0)
        return {};                                   // a plain .usda/.usdc names built-in shaders by id
    SdfZipFile zip = SdfZipFile::Open(p);
    if (! zip)
        return {};

    boost::system::error_code ec;
    const auto        sz  = fs::file_size(p, ec);
    const std::time_t mt  = fs::last_write_time(p, ec);
    const fs::path    dir = fs::temp_directory_path(ec) / "printman_osl"
                          / (fs::path(p).stem().string() + "_" + std::to_string((unsigned long long) sz)
                                                         + "_" + std::to_string((long long) mt));
    bool any = false;
    for (auto it = zip.begin(), e = zip.end(); it != e; ++it) {
        const std::string    fname = *it;
        const std::string::size_type dot = fname.rfind('.');
        const std::string    ext = (dot == std::string::npos) ? std::string() : fname.substr(dot);
        if (ext != ".oso" && ext != ".tx" && ext != ".exr" && ext != ".png" && ext != ".jpg")
            continue;                                // the .usd layer(s) are read by USD from the package
        const SdfZipFile::FileInfo info = it.GetFileInfo();
        if (info.size != info.uncompressedSize)
            continue;                                // usdz entries are stored; skip a compressed one
        const fs::path out = dir / fs::path(fname).filename();
        if (fs::exists(out, ec) && fs::file_size(out, ec) == info.uncompressedSize) { any = true; continue; }
        fs::create_directories(out.parent_path(), ec);
        std::ofstream os(out.string(), std::ios::binary);
        os.write(it.GetFile(), std::streamsize(info.uncompressedSize));
        if (os) any = true;
        else BOOST_LOG_TRIVIAL(error) << "PrintMan: failed to extract bundled shader asset '" << fname
                                      << "' from " << p;
    }
    return any ? dir.string() : std::string();
}

ModelVolume *add_scene_volume(Model *model, ModelObject *object, const std::string &name,
                              const char *path, PrintMan::PrintManScene &&scene)
{
    ModelVolume *volume = object->add_volume(TriangleMesh(scene_proxy_boxes(scene)));
    volume->name              = name;
    volume->source.input_file = path;
    volume->source.object_idx = (int) model->objects.size() - 1;
    volume->source.volume_idx = (int) object->volumes.size() - 1;
    const Vec3d off = volume->source.mesh_offset;
    for (PrintMan::Placement &pl : scene.placements) {
        Transform3d shift = Transform3d::Identity();
        shift.translate(-off);
        pl.xform = shift * pl.xform;
    }
    // Colour de-risk (PRINTMAN_DEBUG_COLOR): declare two filaments so slice_scene emits per-filament
    // contours and the object splits into two extruder regions -- a hardcoded Z-band two-tone, no shader.
    if (std::getenv("PRINTMAN_DEBUG_COLOR"))
        scene.filaments = {1, 2};
    // If this model bundles its shaders/maps (a self-contained .usdz), slice against the extracted copy.
    // Any material -- the mesh's own or a region's -- naming a shader means the bundle may carry assets.
    bool wants_shaders = ! scene.osl_surface_shader.empty() || ! scene.osl_displacement_shader.empty();
    for (const PrintMan::MaterialShaders &mat : scene.extra_materials)
        wants_shaders = wants_shaders || ! mat.surface.empty() || ! mat.displacement.empty();
    if (wants_shaders)
        scene.osl_shader_searchpath = extract_usdz_shader_assets(path);
    volume->printman_scene = std::move(scene);
    return volume;
}

} // namespace

// Merges the whole stage into one mesh. Kept for callers that want geometry
// without Orca's model, and used by the tests.
bool load_usd(const char *path, TriangleMesh *meshptr, std::string &message)
{
    std::vector<NamedMesh> meshes;
    if (!read_stage(path, meshes, message))
        return false;

    indexed_triangle_set merged;
    for (NamedMesh &m : meshes) {
        const int base = int(merged.vertices.size());
        merged.vertices.insert(merged.vertices.end(), m.its.vertices.begin(), m.its.vertices.end());
        for (const stl_triangle_vertex_indices &t : m.its.indices)
            merged.indices.emplace_back(t[0] + base, t[1] + base, t[2] + base);
    }
    *meshptr = TriangleMesh(std::move(merged));
    return true;
}

// One ModelVolume per mesh prim, following Format/STEP.cpp -- a stage's parts
// stay separately selectable, with their own settings, instead of arriving as
// one indivisible lump.
bool load_usd(const char *path, Model *model, std::string &message, const char *object_name_in, bool amplify)
{
    std::string object_name;
    if (object_name_in == nullptr) {
        const char *last_slash = strrchr(path, DIR_SEPARATOR);
        object_name.assign((last_slash == nullptr) ? path : last_slash + 1);
    } else
        object_name.assign(object_name_in);

    // PrintMan amplification: an instanced stage becomes ONE volume carrying the scene beside a
    // proxy bounding-box envelope -- the N-instance geometry is never materialized, so peak
    // memory stays bounded and slice_volume amplifies the scene lazily. amplify=false and
    // non-instanced stages take the flatten path below (tests slice with amplify=false as the
    // trusted transform reference). A stage mixing instances with standalone geometry imports
    // only the instances.
    if (amplify) {
        PrintMan::PrintManScene scene;
        if (build_scene_from_path(path, scene) && ! scene.placements.empty()) {
            ModelObject *object = model->add_object();
            object->name       = object_name;
            object->input_file = path;
            add_scene_volume(model, object, object_name, path, std::move(scene));
            return true;
        }
    }

    // Non-instanced stage (or amplify=false): one ModelVolume per mesh prim; amplify defers cages here too.
    std::vector<NamedMesh> meshes;
    if (! read_stage(path, meshes, message, amplify))
        return false;

    ModelObject *object = model->add_object();
    object->name       = object_name;
    object->input_file = path;

    for (NamedMesh &m : meshes) {
        if (m.cage) {
            // A cage becomes a one-placement PrintMan scene (refined at slice time, cage shown in the
            // viewport); other prims still import flat, so a mixed stage keeps every prim.
            PrintMan::PrintManScene scene;
            scene.prototypes.push_back(std::move(m.its));   // coarse control mesh, prototype-local
            PrintMan::Placement place;
            place.prototype = 0;
            place.xform     = m.xform;
            scene.placements.push_back(place);
            scene.cages.emplace(0, std::move(*m.cage));
            scene.osl_surface_shader      = m.osl_surface_shader;       // material terminals from the prim
            scene.osl_displacement_shader = m.osl_displacement_shader;
            scene.osl_max_displacement    = m.osl_max_displacement;
            scene.color_object_space   = m.color_object_space;   // colour-shader hints from the prim
            scene.color_dither         = m.color_dither;
            scene.color_all_filaments  = m.color_all_filaments;
            scene.osl_surface_params      = std::move(m.surface_params);       // authored shader parameters
            scene.osl_displacement_params = std::move(m.displacement_params);
            scene.extra_materials      = std::move(m.extra_materials);   // region materials (cage->face_material selects)
            add_scene_volume(model, object, m.name, path, std::move(scene));
        } else {
            ModelVolume *volume = object->add_volume(TriangleMesh(std::move(m.its)));
            // The prim path, so a volume in the UI can be traced back to the stage.
            volume->name              = m.name;
            volume->source.input_file = path;
            volume->source.object_idx = (int) model->objects.size() - 1;
            volume->source.volume_idx = (int) object->volumes.size() - 1;
        }
    }

    return true;
}

}; // namespace Slic3r
