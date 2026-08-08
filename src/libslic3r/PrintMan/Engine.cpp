#include "libslic3r/PrintMan/Engine.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

#include "libslic3r/ClipperUtils.hpp"       // union_ex
#include "libslic3r/TriangleMesh.hpp"        // its_flip_triangles
#include "libslic3r/TriangleMeshSlicer.hpp"  // slice_mesh_ex, MeshSlicingParamsEx

namespace Slic3r { namespace PrintMan {

// Conservative world-Z span of a placement (z-range of its box's eight mapped corners), to cull
// the layers it cannot reach.
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

// Bake one placement's prototype into a transient world copy and slice it; the first spanned
// layer goes in `first`, its per-layer sections in `sub`. Pure per-placement work.
static void slice_placement(const PrintManScene &scene, const MeshSlicingParamsEx &params,
                            const std::vector<float> &zs, const Placement &place,
                            const std::function<void()> &throw_on_cancel,
                            size_t &first, std::vector<ExPolygons> &sub)
{
    first = 0;
    sub.clear();
    if (place.prototype < 0 || size_t(place.prototype) >= scene.prototypes.size())
        return;
    const indexed_triangle_set &proto = scene.prototypes[place.prototype];
    if (proto.empty())
        return;

    const Transform3d m = params.trafo * place.xform;

    double zmin = 0., zmax = 0.;
    placement_z_span(proto, m, zmin, zmax);
    first = size_t(std::lower_bound(zs.begin(), zs.end(), float(zmin)) - zs.begin());
    const size_t last = size_t(std::upper_bound(zs.begin(), zs.end(), float(zmax)) - zs.begin());
    if (first >= last)
        return;
    const std::vector<float> sub_zs(zs.begin() + first, zs.begin() + last);

    MeshSlicingParamsEx p = params;
    p.trafo = Transform3d::Identity();
    p.slicing_mode_normal_below_layer = params.slicing_mode_normal_below_layer > first
        ? params.slicing_mode_normal_below_layer - first : 0;

    // Bake to world and slice with identity trafo, rather than slicing through a rotating
    // params.trafo: make_trafo_for_slicing prescales XY only, so a tilted placement (XY mixed
    // with Z) would skew the cross-sections. Baking first is what the naive flatten does.
    indexed_triangle_set world = proto;
    for (Vec3f &v : world.vertices)
        v = (m * v.cast<double>()).cast<float>();
    if (m.linear().determinant() < 0.)   // mirror / negative scale: flip faces so winding stays outward
        its_flip_triangles(world);
    sub = slice_mesh_ex(world, sub_zs, p, throw_on_cancel);
}

std::vector<ExPolygons> slice_scene(
    const PrintManScene         &scene,
    const MeshSlicingParamsEx   &params,
    const std::vector<float>    &zs,
    const std::function<void()> &throw_on_cancel,
    const std::function<void(size_t, size_t)> &report_progress)
{
    // Slice placements in parallel into thread-local per-layer buckets, merged at the end. Union
    // is associative and the nonzero rule idempotent, so the result matches serial order. Buckets
    // collapse at kBucketCap, bounding peak memory to ~(workers * kBucketCap) sections per layer.
    constexpr size_t kBucketCap = 64;
    const size_t     nplace     = scene.placements.size();

    tbb::enumerable_thread_specific<std::vector<ExPolygons>> tls(
        [&zs]{ return std::vector<ExPolygons>(zs.size()); });
    // report_progress reaches Orca's status callback, which is not concurrency-safe, so serialize.
    std::atomic<size_t> done{0};
    std::mutex          report_mtx;
    const size_t report_step = report_progress ? std::max<size_t>(size_t(1), nplace / 100) : 0;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, nplace),
        [&](const tbb::blocked_range<size_t> &range) {
            std::vector<ExPolygons> &out_local = tls.local();
            size_t                   first = 0;
            std::vector<ExPolygons>  sub;
            for (size_t pi = range.begin(); pi < range.end(); ++ pi) {
                slice_placement(scene, params, zs, scene.placements[pi], throw_on_cancel, first, sub);
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
                if (report_step) {
                    const size_t d = ++ done;
                    if (d % report_step == 0) {
                        std::lock_guard<std::mutex> lock(report_mtx);
                        report_progress(d, nplace);
                    }
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

    return out;
}

}} // namespace Slic3r::PrintMan
