#include "libslic3r/PrintMan/Engine.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <utility>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>

#include <Eigen/Dense>
#include <boost/log/trivial.hpp>

#include "libslic3r/ClipperUtils.hpp"       // union_ex
#include "libslic3r/TriangleMesh.hpp"        // its_flip_triangles
#include "libslic3r/TriangleMeshSlicer.hpp"  // slice_mesh_ex, MeshSlicingParamsEx
#include "libslic3r/Format/USDSubdiv.hpp"    // build_cage, subdivide, refine_region, device_level
#include "libslic3r/PrintMan/Palette.hpp"    // nearest_filament (colour quantization)
#include "libslic3r/PrintMan/CoreAdapter.hpp" // vendored core: amplify_subcage_adaptive + host<->core converters

namespace Slic3r { namespace PrintMan {

// Fixed level for loop/bilinear cages (whole-cage path) and the import proxy AABB; Catmull-Clark
// cages use the per-placement device level (slice_cage_placement).
static constexpr int kSubdivLevel = 2;

// A cage is sliced in Z-bands: only faces whose limit reaches a band are refined, so peak geometry is
// one band, not the whole surface. Band thickness ~ a face's 1-ring Z-span, clamped to this range.
static constexpr size_t kMinLayersPerBand = 16;
static constexpr size_t kMaxLayersPerBand = 256;

// Sections per layer at which a bucket is unioned down, bounding peak memory to ~(workers * cap).
static constexpr size_t kBucketCap = 64;

// Fan-triangulate a refined n-gon cage, optional winding flip for a leftHanded prim (the world/mirror
// flip is left to the caller, as for a plain prototype). When the cage carries face-varying UV and tri_uv
// is given, it is filled with one (u,v) per emitted triangle -- the mean of that triangle's face-corner UVs
// (the winding flip does not change the mean), parallel to its.indices, for per-face colour sampling.
static indexed_triangle_set triangulate_cage(const usd_subdiv::Cage &c, bool flip,
                                             std::vector<std::array<float, 2>> *tri_uv = nullptr)
{
    indexed_triangle_set its;
    its.vertices.reserve(c.verts.size());
    for (const usd_subdiv::Pt &p : c.verts)
        its.vertices.emplace_back(float(p[0]), float(p[1]), float(p[2]));
    const bool uv = c.has_uv() && tri_uv;
    for (int f = 0; f < c.nfaces(); ++ f) {
        const int off = c.foff[f];
        const int n   = c.foff[f + 1] - off;
        for (int k = 1; k + 1 < n; ++ k) {
            int a = c.fvi[off], b = c.fvi[off + k], d = c.fvi[off + k + 1];
            if (flip) std::swap(b, d);
            its.indices.emplace_back(a, b, d);
            if (uv) {
                const auto &u0 = c.fvar[off], &u1 = c.fvar[off + k], &u2 = c.fvar[off + k + 1];
                tri_uv->push_back({float((u0[0] + u1[0] + u2[0]) / 3.0), float((u0[1] + u1[1] + u2[1]) / 3.0)});
            }
        }
    }
    return its;
}

// World-ready mesh per plain prototype. A catmullClark cage is left empty (refined lazily per Z-band
// in slice_cage_placement); loop/bilinear cages are refined whole here, as the band path is CC only.
static std::vector<indexed_triangle_set> materialize_prototypes(const PrintManScene &scene)
{
    std::vector<indexed_triangle_set> out(scene.prototypes.size());
    for (size_t i = 0; i < scene.prototypes.size(); ++ i) {
        auto it = scene.cages.find(int(i));
        if (it == scene.cages.end()) {
            out[i] = scene.prototypes[i];
            continue;
        }
        const SubdivCage &sc = it->second;
        if (sc.scheme == "catmullClark")
            continue;                                    // refined lazily per Z-band
        usd_subdiv::Cage cage = usd_subdiv::build_cage(
            sc.points, sc.face_counts, sc.face_indices,
            sc.crease_indices, sc.crease_lengths, sc.crease_sharp,
            sc.corner_indices, sc.corner_sharp, sc.boundary, sc.triangle_smooth);
        cage = usd_subdiv::subdivide(cage, sc.scheme, kSubdivLevel);
        out[i] = triangulate_cage(cage, sc.flip_winding);
    }
    return out;
}

// A subdivision cage carried to slice time: the built control cage plus the adjacency the band loop
// needs, once per prototype (subdivision is affine-covariant, so it is refined in local space).
struct CageProto {
    usd_subdiv::Cage              cage;       // control cage (unrefined), local frame
    std::vector<std::vector<int>> vf;         // vertex -> incident control faces
    std::vector<std::vector<int>> ring_vids;  // face -> its 1-ring control vertices (limit-patch support)
    std::string                   scheme;     // catmullClark (only scheme the band path refines)
    bool                          flip_winding = false;
    // device_level inputs: longest control edge (local) and largest crease sharpness; sigma per placement.
    double                        max_edge = 0.0;
    double                        max_crease = 0.0;
};

// Build a CageProto for each subdivision-cage prototype; plain prototypes get no descriptor.
static std::vector<std::optional<CageProto>> build_cage_protos(const PrintManScene &scene)
{
    std::vector<std::optional<CageProto>> out(scene.prototypes.size());
    for (const auto &kv : scene.cages) {
        const int i = kv.first;
        if (i < 0 || size_t(i) >= out.size())
            continue;
        const SubdivCage &sc = kv.second;
        if (sc.scheme != "catmullClark")
            continue;                                    // loop / bilinear slice eagerly (plain path)
        CageProto cp;
        cp.cage = usd_subdiv::build_cage(
            sc.points, sc.face_counts, sc.face_indices,
            sc.crease_indices, sc.crease_lengths, sc.crease_sharp,
            sc.corner_indices, sc.corner_sharp, sc.boundary, sc.triangle_smooth, sc.st);
        cp.vf = usd_subdiv::vertex_faces(cp.cage);
        cp.ring_vids.resize(cp.cage.nfaces());
        for (int f = 0; f < cp.cage.nfaces(); ++ f) {
            std::vector<int> &rv = cp.ring_vids[f];
            for (int k = cp.cage.foff[f]; k < cp.cage.foff[f + 1]; ++ k)
                for (int nf : cp.vf[cp.cage.fvi[k]])
                    for (int kk = cp.cage.foff[nf]; kk < cp.cage.foff[nf + 1]; ++ kk)
                        rv.push_back(cp.cage.fvi[kk]);
            std::sort(rv.begin(), rv.end());
            rv.erase(std::unique(rv.begin(), rv.end()), rv.end());
        }
        cp.scheme            = sc.scheme;
        cp.flip_winding      = sc.flip_winding;
        cp.max_edge          = usd_subdiv::max_control_edge(cp.cage);
        cp.max_crease        = usd_subdiv::max_crease_sharpness(cp.cage);
        out[i]               = std::move(cp);
    }
    return out;
}

// Conservative world-Z span of a placement (its box's eight mapped corners), to cull unreachable layers.
static void placement_z_span(const indexed_triangle_set &its, const Transform3d &m,
                             double &zmin, double &zmax)
{
    Vec3f lo = its.vertices.front();
    Vec3f hi = lo;
    for (const Vec3f &v : its.vertices) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
    zmin =  std::numeric_limits<double>::infinity();
    zmax = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < 8; ++ i) {
        Vec3d c((i & 1) ? hi.x() : lo.x(),
                (i & 2) ? hi.y() : lo.y(),
                (i & 4) ? hi.z() : lo.z());
        double z = (m * c).z();
        zmin = std::min(zmin, z);
        zmax = std::max(zmax, z);
    }
}

// Merge per-layer sections into the thread-local buckets from layer `first`. Union is associative and
// idempotent, so arrival order and the kBucketCap collapse are safe. `sub` is consumed.
static void accumulate_sections(std::vector<ExPolygons> &out_local, size_t first,
                                std::vector<ExPolygons> &sub)
{
    for (size_t i = 0; i < sub.size(); ++ i) {
        if (sub[i].empty())
            continue;
        ExPolygons &bucket = out_local[first + i];
        if (bucket.empty()) {
            bucket = std::move(sub[i]);
        } else {
            for (ExPolygon &ep : sub[i])
                bucket.emplace_back(std::move(ep));
            if (bucket.size() >= kBucketCap)
                bucket = union_ex(bucket);
        }
    }
}

// Bake one plain-prototype placement to a world copy and slice it into `out_local`.
static void slice_placement(const std::vector<indexed_triangle_set> &protos,
                            const MeshSlicingParamsEx &params,
                            const std::vector<float> &zs, const Placement &place,
                            const std::function<void()> &throw_on_cancel,
                            std::vector<ExPolygons> &out_local)
{
    if (place.prototype < 0 || size_t(place.prototype) >= protos.size())
        return;
    const indexed_triangle_set &proto = protos[place.prototype];
    if (proto.empty())
        return;

    const Transform3d m = params.trafo * place.xform;

    double zmin = 0., zmax = 0.;
    placement_z_span(proto, m, zmin, zmax);
    const size_t first = size_t(std::lower_bound(zs.begin(), zs.end(), float(zmin)) - zs.begin());
    const size_t last  = size_t(std::upper_bound(zs.begin(), zs.end(), float(zmax)) - zs.begin());
    if (first >= last)
        return;
    const std::vector<float> sub_zs(zs.begin() + first, zs.begin() + last);

    MeshSlicingParamsEx p = params;
    p.trafo = Transform3d::Identity();
    p.slicing_mode_normal_below_layer = params.slicing_mode_normal_below_layer > first
        ? params.slicing_mode_normal_below_layer - first : 0;

    // Bake to world and slice with identity trafo: slicing through params.trafo prescales XY only, so a
    // tilted placement (XY mixed with Z) would skew the cross-sections. Baking first matches the flatten.
    indexed_triangle_set world = proto;
    for (Vec3f &v : world.vertices)
        v = (m * v.cast<double>()).cast<float>();
    if (m.linear().determinant() < 0.)   // mirror / negative scale: flip faces so winding stays outward
        its_flip_triangles(world);
    std::vector<ExPolygons> sub = slice_mesh_ex(world, sub_zs, p, throw_on_cancel);
    accumulate_sections(out_local, first, sub);
}

// Drop whole triangles whose world-Z range is entirely outside [zlo, zhi] -- a ragged boundary, not a
// geometric clip, so every triangle reaching the band is kept and the cross-sections stay closed.
static void drop_triangles_outside_band(indexed_triangle_set &its, double zlo, double zhi)
{
    decltype(its.indices) kept;
    kept.reserve(its.indices.size());
    for (const auto &t : its.indices) {
        const double a = its.vertices[t[0]].z(), b = its.vertices[t[1]].z(), c = its.vertices[t[2]].z();
        if (std::max({a, b, c}) >= zlo && std::min({a, b, c}) <= zhi)
            kept.push_back(t);
    }
    its.indices.swap(kept);
}

// Debug guard: after dropping, every boundary edge must lie strictly outside [zlo, zhi], so no band
// cross-section is open. It matters because slice_mesh_ex silently heals open contours up to a 2 mm
// chord, so a coverage miss would be wrong-but-quiet. True by construction on manifold input (a face's
// children never leave its 1-ring hull); assert only -- compiled out under NDEBUG.
[[maybe_unused]] static bool band_is_closed_over(const indexed_triangle_set &its, double zlo, double zhi)
{
    std::map<std::pair<int, int>, int> edge_count;
    for (const auto &t : its.indices)
        for (int e = 0; e < 3; ++ e) {
            int a = t[e], b = t[(e + 1) % 3];
            edge_count[{std::min(a, b), std::max(a, b)}]++;
        }
    for (const auto &kv : edge_count) {
        if (kv.second != 1)
            continue;   // interior (shared) edge
        const double za = its.vertices[kv.first.first].z(), zb = its.vertices[kv.first.second].z();
        if (std::max(za, zb) >= zlo && std::min(za, zb) <= zhi)
            return false;   // a boundary edge reaches into the sliced band
    }
    return true;
}

// Per-face 1-ring-hull world-Z ranges (flo/fhi, the band selection key) and the widest span (max_span,
// band sizing). Shared by the slicing loop and the progress pre-pass so their band boundaries agree.
namespace { struct BandKeys { std::vector<double> flo, fhi; double max_span = 0.0; }; }

static BandKeys cage_band_keys(const CageProto &cp, const Transform3d &m)
{
    const usd_subdiv::Cage &cage = cp.cage;
    std::vector<double> wz(cage.nverts());
    for (int v = 0; v < cage.nverts(); ++ v)
        wz[v] = (m * Vec3d(cage.verts[v][0], cage.verts[v][1], cage.verts[v][2])).z();
    BandKeys k;
    k.flo.resize(cage.nfaces());
    k.fhi.resize(cage.nfaces());
    for (int f = 0; f < cage.nfaces(); ++ f) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (int v : cp.ring_vids[f]) { lo = std::min(lo, wz[v]); hi = std::max(hi, wz[v]); }
        k.flo[f] = lo; k.fhi[f] = hi;
        k.max_span = std::max(k.max_span, hi - lo);
    }
    return k;
}

// Z-band thickness in layers for a cage placement (rationale at the call in slice_cage_placement).
static size_t band_layers_for(double max_span, const std::vector<float> &zs)
{
    const size_t nlayers = zs.size();
    const double layer_h = nlayers > 1 ? double(zs[1] - zs[0]) : 0.0;
    return std::min(kMaxLayersPerBand, std::max(kMinLayersPerBand,
        layer_h > 0.0 ? size_t(std::ceil(max_span / layer_h)) : kMinLayersPerBand));
}

// Number of Z-bands slice_cage_placement will process -- the per-band progress unit. Matches the band
// loop's iteration count exactly.
static size_t cage_band_count(const CageProto &cp, const MeshSlicingParamsEx &params,
                              const std::vector<float> &zs, const Placement &place)
{
    if (cp.cage.nfaces() == 0 || zs.empty())
        return 0;
    const Transform3d m  = params.trafo * place.xform;
    const size_t band_layers = band_layers_for(cage_band_keys(cp, m).max_span, zs);
    return (zs.size() + band_layers - 1) / band_layers;
}

// Colour classification (Phase B): deposit a colour ribbon along each refined face's slice segment.
// A face is quantized to a filament channel (nearest_filament by its centroid Cout); on every layer it
// crosses, its triangle-plane intersection is a short segment, thickened to `band_width` into a thin
// quad and appended to that layer/channel. The band owns the layer rows [first, last), so its writes
// into `bands` are disjoint from the other bands of the same placement. slice_scene unions each
// channel's ribbons and clips them to the layer contour -- so the wall is coloured by its surface,
// within a single layer, not just voted whole (that was Phase A).
static void accumulate_color_bands(const indexed_triangle_set &its, const std::vector<float> &zs,
                                   size_t first, size_t last, const ColorField &color,
                                   const std::vector<std::array<float, 2>> &tri_uv,
                                   std::vector<std::vector<Polygons>> &bands)
{
    const double half     = 0.5 * color.band_width;
    const bool   have_uv  = tri_uv.size() == its.indices.size();   // authored UV, one (u,v) per triangle
    for (size_t ti = 0; ti < its.indices.size(); ++ ti) {
        const auto  &t = its.indices[ti];
        const Vec3f &a = its.vertices[t[0]], &b = its.vertices[t[1]], &c = its.vertices[t[2]];
        const V3 centroid{{(double(a.x()) + b.x() + c.x()) / 3.0,
                           (double(a.y()) + b.y() + c.y()) / 3.0,
                           (double(a.z()) + b.z() + c.z()) / 3.0}};
        const V3 e1{{double(b.x()) - a.x(), double(b.y()) - a.y(), double(b.z()) - a.z()}};
        const V3 e2{{double(c.x()) - a.x(), double(c.y()) - a.y(), double(c.z()) - a.z()}};
        const double su = have_uv ? tri_uv[ti][0] : 0.0;
        const double sv = have_uv ? tri_uv[ti][1] : 0.0;
        const V3  cout = color.eval(centroid, face_normal(a, b, c), e1, e2, su, sv);
        const int k    = color.dither ? dither_filament(cout, color.palette, dither_hash(centroid, color.dither_cell))
                                      : nearest_filament(cout, color.palette);
        if (k < 0)
            continue;
        const float        zlo  = std::min({a.z(), b.z(), c.z()}), zhi = std::max({a.z(), b.z(), c.z()});
        const Vec3f *const v[3] = {&a, &b, &c};
        for (size_t L = first; L < last; ++ L) {
            const float z = zs[L];
            if (z < zlo || z > zhi)
                continue;
            // A triangle crossing a plane meets it in a segment: the two edges with endpoints on
            // opposite sides each contribute one crossing point (interpolated in XY).
            Vec2d p[2];
            int   n = 0;
            for (int e = 0; e < 3 && n < 2; ++ e) {
                const Vec3f &u = *v[e], &w = *v[(e + 1) % 3];
                if ((u.z() - z) * (w.z() - z) < 0.0f) {
                    const double s = double(z - u.z()) / double(w.z() - u.z());
                    p[n++] = Vec2d(u.x() + s * (w.x() - u.x()), u.y() + s * (w.y() - u.y()));
                }
            }
            if (n < 2)
                continue;   // a vertex sits exactly on the plane or the face is coplanar (rare on a
                            // curved wall): skipped -- may leave a sub-perimeter sliver uncoloured
            const Vec2d  dir = p[1] - p[0];
            const double len = dir.norm();
            if (len < 1e-9)
                continue;
            const Vec2d o(-dir.y() / len * half, dir.x() / len * half);   // perpendicular offset, `half` mm
            Polygon quad;
            quad.points = {Point(scaled<coord_t>(p[0].x() + o.x()), scaled<coord_t>(p[0].y() + o.y())),
                           Point(scaled<coord_t>(p[1].x() + o.x()), scaled<coord_t>(p[1].y() + o.y())),
                           Point(scaled<coord_t>(p[1].x() - o.x()), scaled<coord_t>(p[1].y() - o.y())),
                           Point(scaled<coord_t>(p[0].x() - o.x()), scaled<coord_t>(p[0].y() - o.y()))};
            bands[L][size_t(k)].emplace_back(std::move(quad));
        }
    }
}

// Slice one placement of a subdivision cage in Z-bands. The cage is refined to a device-perfect level
// for this placement's world scale. Per band, only faces whose limit reaches it are refined -- a
// Catmull-Clark child never leaves its face's 1-ring hull, so a face is selected by that hull's
// world-Z range. Each output layer lies in exactly one band.
static void slice_cage_placement(const CageProto &cp,
                                 const MeshSlicingParamsEx &params,
                                 const std::vector<float> &zs, const Placement &place,
                                 const std::function<void()> &throw_on_cancel,
                                 std::vector<ExPolygons> &out_local,
                                 const std::function<void()> &on_band,
                                 const DisplacementField &displacement,
                                 const ColorField &color,
                                 std::vector<std::vector<Polygons>> *bands_local)
{
    const usd_subdiv::Cage &cage = cp.cage;
    assert(cp.scheme == "catmullClark");   // build_cage_protos engages a CageProto only for CC cages
    if (cage.nfaces() == 0)
        return;

    const Transform3d m        = params.trafo * place.xform;
    const bool        flip_det = m.linear().determinant() < 0.;

    // Device-perfect level for this placement. sigma (largest singular value of the world linear map)
    // scales a local edge to world; subdiv_tol is the device rate, 0 skips the size term and the SVD.
    double sigma = 1.0;
    if (params.subdiv_tol > 0.0)
        sigma = Eigen::JacobiSVD<Matrix3d>(Matrix3d(m.linear())).singularValues()(0);
    bool capped = false;
    const int level = usd_subdiv::device_level(sigma, cp.max_edge, params.subdiv_tol,
                                               cp.max_crease, &capped);
    if (capped)
        BOOST_LOG_TRIVIAL(warning)
            << "PrintMan: a subdivision cage needs more than the level-8 refinement cap at this"
               " device rate; its highest-curvature region may print slightly faceted.";

    // Each face's 1-ring-hull world-Z range (the selection key) and the widest span (band sizing).
    const BandKeys bk = cage_band_keys(cp, m);

    // Displacement moves refined vertices along their normal by up to this, so the band's face
    // selection is grown by it below. Two first-cut limits (see the PrintMan design docs):
    //   - Orca's layer set is fixed before we run, so relief that pushes past the top/bottom
    //     layers is clipped.
    //   - SEAM CONTINUITY across bands holds only while max_disp is comparable to the layer
    //     height. Each band computes vertex normals on its core-ONLY refined mesh, so a shared
    //     boundary vertex can get slightly different normals in adjacent bands. When max_disp is
    //     >~ a layer, the grown selection keeps every incident face co-present in both bands and
    //     the normals agree; a low-amplitude field (max_disp < layer height) can step quietly at
    //     a band boundary. The fix -- band-invariant (halo-consistent / analytic-limit) normals --
    //     is a follow-up; until then displacement is only sound for max_disp >~ the layer height.
    const double max_disp = displacement ? displacement.max_magnitude : 0.0;

    // Band thickness ~ the widest 1-ring Z-span (in layers), clamped: a face then falls in ~one band,
    // not every band its 1-ring touches -- a memory-for-refine trade.
    const size_t nlayers     = zs.size();
    const size_t band_layers = band_layers_for(bk.max_span, zs);

    std::vector<std::pair<size_t, size_t>> band_ranges;   // [first, last) per band, disjoint in layers
    for (size_t first = 0; first < nlayers; first += band_layers)
        band_ranges.push_back({first, std::min(first + band_layers, nlayers)});

    // Fold: dice + displace through the vendored standalone engine (adaptive per-face, crack-free, any
    // cage) instead of the fork's uniform refine_region + apply_displacement. The world-space core cage
    // is built once (m baked in; winding reversed for the cage's authored flip XOR a det<0 mirror, so
    // the winding-derived normals stay outward); a BridgeShader forwards the DisplacementField.
    const bool                                  net_reverse = cp.flip_winding != flip_det;
    const printman::subdiv::Cage                core_cage   = to_world_core_cage(cage, m, net_reverse);
    BridgeShader                                bridge(&displacement, max_disp);
    const std::vector<const printman::Shader *> core_shaders{&bridge};

    // Slice one band: the core dices + displaces this band's faces (for its own layer subset), then the
    // host colours + slices the returned mesh. Bands are independent: each writes only its own (disjoint)
    // layer range of out_local, so they run in parallel.
    const auto slice_band = [&](size_t first, size_t last) {
        std::vector<double> band_zs_d(zs.begin() + first, zs.begin() + last);   // this band's layers (ascending)

        // The core selects this band's faces (grown by max_disp), dices them adaptively, displaces via
        // the bridged shader, and hands back one band mesh (nbands=1 -- the host owns the band split
        // here). do_slice=false: the host colours + slices the mesh with slice_mesh_ex below.
        const auto on_core_band = [&](const printman::Mesh &bm) {
            if (bm.tri.empty())
                return;
            indexed_triangle_set its;
            append_core_mesh_to_its(bm, its);
            // An out-of-band triangle crosses none of this band's layers, so it deposits no colour and
            // slices to nothing -- the band mesh spans only this band's z (plus the max_disp halo).
            if (color && bands_local) {
                const std::vector<std::array<float, 2>> tri_uv = tri_uv_from_core_mesh(bm);
                accumulate_color_bands(its, zs, first, last, color, tri_uv, *bands_local);
            }
            MeshSlicingParamsEx p = params;
            p.trafo = Transform3d::Identity();
            p.slicing_mode_normal_below_layer = params.slicing_mode_normal_below_layer > first
                ? params.slicing_mode_normal_below_layer - first : 0;
            const std::vector<float> band_zs(zs.begin() + first, zs.begin() + last);
            std::vector<ExPolygons>  sub = slice_mesh_ex(its, band_zs, p, throw_on_cancel);
            accumulate_sections(out_local, first, sub);
            throw_on_cancel();
        };
        printman::amplify_subcage_adaptive(core_cage, /*ctag*/ {}, core_shaders, level, params.subdiv_tol,
                                           band_zs_d, /*nbands*/ 1, on_core_band, /*do_slice*/ false);
        if (on_band) on_band();   // one band done -- advance the status bar (see slice_scene)
    };

    // Slice all bands in one parallel_for so TBB load-balances the whole pool (bands are very uneven).
    // Peak live bands is now bounded by the worker count, not a fixed cap -- a memory/throughput trade.
    // isolate() is required: out_local is a thread-keyed ETS slot; without it a blocked worker could
    // steal another placement's outer task and race this placement's bands.
    tbb::this_task_arena::isolate([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, band_ranges.size(), 1),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t bi = r.begin(); bi < r.end(); ++ bi)
                    slice_band(band_ranges[bi].first, band_ranges[bi].second);
            });
    });

    // The band's face selection was grown by the declared max_disp; if the shader actually moved a vertex
    // farther, relief past the grown band can be clipped by Orca's fixed layer set. Warn so the user raises
    // printman:maxMagnitude (restores the over-bound check the pre-fold apply_displacement path had).
    const double peak = bridge.peak.load(std::memory_order_relaxed);
    if (max_disp > 0.0 && peak > max_disp + 1e-6)
        BOOST_LOG_TRIVIAL(warning)
            << "PrintMan: displacement reached " << peak << " mm, past the declared maxMagnitude "
            << max_disp << " mm; relief beyond the grown slice band may be clipped -- raise"
               " printman:maxMagnitude.";
}

std::vector<ExPolygons> slice_scene(
    const PrintManScene         &scene,
    const MeshSlicingParamsEx   &params,
    const std::vector<float>    &zs,
    const std::function<void()> &throw_on_cancel,
    const std::function<void(size_t, size_t)> &report_progress,
    const DisplacementField     &displacement,
    const ColorField            &color,
    std::vector<std::vector<ExPolygons>> *out_segmentation)
{
    // Slice placements in parallel into thread-local per-layer buckets, merged at the end. Union is
    // associative and idempotent, so the result matches serial order.
    const size_t nplace = scene.placements.size();

    // Plain prototypes as world-ready meshes; cages as descriptors refined per band. Built once, shared.
    const std::vector<indexed_triangle_set>    protos     = materialize_prototypes(scene);
    const std::vector<std::optional<CageProto>> cageprotos = build_cage_protos(scene);

    // Progress counted in Z-bands (a cage = its band count, a plain mesh = 1 unit), so the bar advances
    // within a single large cage; total precomputed so the denominator is fixed.
    auto cage_at = [&](size_t pi) -> const CageProto* {
        const int proto = scene.placements[pi].prototype;
        return (proto >= 0 && size_t(proto) < cageprotos.size() && cageprotos[proto])
            ? &*cageprotos[proto] : nullptr;
    };
    size_t total_units = 0;
    if (report_progress)
        for (size_t pi = 0; pi < nplace; ++ pi) {
            if (const CageProto *cp = cage_at(pi))
                total_units += cage_band_count(*cp, params, zs, scene.placements[pi]);
            else
                ++ total_units;
        }

    tbb::enumerable_thread_specific<std::vector<ExPolygons>> tls(
        [&zs]{ return std::vector<ExPolygons>(zs.size()); });
    // report_progress is not concurrency-safe, so serialize it; report at ~1% to keep it off the hot loop.
    std::atomic<size_t> done{0};
    std::mutex          report_mtx;
    const size_t report_step = report_progress ? std::max<size_t>(size_t(1), total_units / 100) : 0;
    const std::function<void()> bump = [&]{
        if (! report_step) return;
        const size_t d = ++ done;
        if (d % report_step == 0) {
            std::lock_guard<std::mutex> lock(report_mtx);
            report_progress(d, total_units);
        }
    };

    // Colour (Phase B): per-layer per-channel ribbon quads, thread-local like the contours and merged
    // into out_segmentation below. Only built when both a ColorField and an out sink are given;
    // plain-mesh placements are not classified (colour targets the amplified cage surface).
    const bool   do_color = bool(color) && out_segmentation != nullptr;
    const size_t npal     = color.palette.size();
    tbb::enumerable_thread_specific<std::vector<std::vector<Polygons>>> tls_bands(
        [&]{ return std::vector<std::vector<Polygons>>(zs.size(), std::vector<Polygons>(npal)); });

    // Slice each placement: a cage through slice_cage_placement (Z-bands), a plain mesh through
    // slice_placement. A single large cage's parallelism comes from its bands; an instanced scene's
    // from the placements themselves (nested band parallelism then just fills any remaining cores).
    tbb::parallel_for(tbb::blocked_range<size_t>(0, nplace),
        [&](const tbb::blocked_range<size_t> &range) {
            std::vector<ExPolygons>            &out_local = tls.local();
            std::vector<std::vector<Polygons>> *bl        = do_color ? &tls_bands.local() : nullptr;
            for (size_t pi = range.begin(); pi < range.end(); ++ pi) {
                const Placement &place = scene.placements[pi];
                const int        proto = place.prototype;
                if (proto >= 0 && size_t(proto) < cageprotos.size() && cageprotos[proto])
                    slice_cage_placement(*cageprotos[proto], params, zs, place, throw_on_cancel, out_local, bump, displacement, color, bl);
                else {
                    slice_placement(protos, params, zs, place, throw_on_cancel, out_local);
                    bump();
                }
                throw_on_cancel();
            }
        });

    std::vector<ExPolygons> out(zs.size());
    for (std::vector<ExPolygons> &out_local : tls)
        for (size_t L = 0; L < out.size(); ++ L) {
            if (out_local[L].empty())
                continue;
            if (out[L].empty())
                out[L] = std::move(out_local[L]);
            else
                for (ExPolygon &ep : out_local[L])
                    out[L].emplace_back(std::move(ep));
        }

    // Collapse each bucket's remaining sections in one parallel pass; a lone section is already a region.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, out.size()),
        [&out, &throw_on_cancel](const tbb::blocked_range<size_t> &range) {
            for (size_t L = range.begin(); L < range.end(); ++ L)
                if (out[L].size() > 1)
                    out[L] = union_ex(out[L]);
            throw_on_cancel();
        });

    // Colour (Phase B): give each layer's outer-wall shell to the surface colour. Each channel's per-face
    // ribbons are the colour VOTES. With wall_depth set, the shell (contour minus its inward offset by one
    // wall depth -- the ring the printer extrudes as perimeters) is partitioned among the channels by nearest
    // vote, so a claim spans the full wall depth (always >= one line wide, on flat or curved geometry) while
    // keeping its tangential extent (the colour proportions). The base filament keeps only the hidden interior
    // and any shell it genuinely voted. Without wall_depth the legacy thin-ribbon clip is used (fragile on a
    // curve; a plain-mesh-only layer stays unassigned either way).
    if (do_color) {
        out_segmentation->assign(zs.size(), std::vector<ExPolygons>(npal));
        std::vector<std::vector<Polygons>> bands(zs.size(), std::vector<Polygons>(npal));
        for (std::vector<std::vector<Polygons>> &bloc : tls_bands)
            for (size_t L = 0; L < zs.size(); ++ L)
                for (size_t k = 0; k < npal; ++ k)
                    for (Polygon &poly : bloc[L][k])
                        bands[L][k].emplace_back(std::move(poly));
        const double wall_depth = color.wall_depth;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, zs.size()),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t L = r.begin(); L < r.end(); ++ L) {
                    if (out[L].empty())
                        continue;
                    if (wall_depth <= 0.0) {
                        for (size_t k = 0; k < npal; ++ k)
                            if (! bands[L][k].empty())
                                (*out_segmentation)[L][k] = intersection_ex(union_ex(bands[L][k]), out[L]);
                        throw_on_cancel();
                        continue;
                    }
                    const ExPolygons interior = offset_ex(out[L], - scaled<float>(wall_depth));
                    const ExPolygons shell     = interior.empty() ? out[L] : diff_ex(out[L], interior);
                    // Each channel claims the shell behind its votes, grown inward by one wall depth and taken in
                    // order so the claims stay disjoint (apply_segmentation steals each extruder against the
                    // unmodified base, so overlaps would double-assign). One offset + one diff per channel; a
                    // colour boundary can shift up to ~wall_depth toward the lower-index filament (acceptable).
                    ExPolygons remaining = shell;
                    for (size_t k = 0; k < npal && ! remaining.empty(); ++ k) {
                        if (bands[L][k].empty())
                            continue;
                        const ExPolygons vote = intersection_ex(union_ex(bands[L][k]), shell);
                        if (vote.empty())
                            continue;
                        ExPolygons claim = intersection_ex(offset_ex(vote, scaled<float>(wall_depth), jtSquare), remaining);
                        if (claim.empty())
                            continue;
                        remaining = diff_ex(remaining, claim);
                        (*out_segmentation)[L][k] = std::move(claim);
                    }
                    throw_on_cancel();
                }
            });
    }

    return out;
}

}} // namespace Slic3r::PrintMan
