#ifndef slic3r_PrintMan_Engine_hpp_
#define slic3r_PrintMan_Engine_hpp_

#include <functional>
#include <vector>

#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/PrintMan/PrintManScene.hpp"

namespace Slic3r {
struct MeshSlicingParamsEx;   // libslic3r/TriangleMeshSlicer.hpp
namespace PrintMan {

// Slice an instanced scene into per-layer contours by amplifying one prototype through each
// placement, unioned -- the N-instance geometry is never baked. Mirrors slice_mesh_ex so it
// stands in at the slice_volume seam. Divergences from slicing one combined mesh: vase mode
// keeps N largest contours, and closing_radius closes each instance before the union.
std::vector<ExPolygons> slice_scene(
    const PrintManScene         &scene,
    const MeshSlicingParamsEx   &params,
    const std::vector<float>    &zs,
    const std::function<void()> &throw_on_cancel = [](){},
    const std::function<void(size_t, size_t)> &report_progress = {});

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_Engine_hpp_
