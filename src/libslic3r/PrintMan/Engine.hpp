#ifndef slic3r_PrintMan_Engine_hpp_
#define slic3r_PrintMan_Engine_hpp_

#include <functional>
#include <vector>

#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/FlushVolPredictor.hpp"   // FlushPredict::RGBColor
#include "libslic3r/PrintMan/PrintManScene.hpp"
#include "libslic3r/PrintMan/Displace.hpp"   // DisplaceShader, V3

namespace Slic3r {
struct MeshSlicingParamsEx;   // libslic3r/TriangleMeshSlicer.hpp
namespace PrintMan {

// Optional displacement shader for the amplification engine: it moves each refined surface
// vertex along its normal by eval(world_point, unit_normal) mm. max_magnitude bounds |eval| so
// the band loop can grow its face selection. Empty (the default) leaves the existing path.
struct DisplacementField {
    DisplaceShader  eval;      // (point, normal) -- built-in C++ shaders
    DisplaceShaderD eval_d;    // (point, normal, dPdx, dPdy) -- footprint-aware (OSL); preferred if set
    double          max_magnitude = 0.0;
    explicit operator bool() const { return bool(eval) || bool(eval_d); }
};

// Optional surface-colour shader for the amplification engine: eval(point, normal, dPdx, dPdy, u, v) is the
// linear-RGB colour of a refined surface point, quantized to `palette` (the loaded filament colours as
// sRGB bytes; channel k = palette[k]). (u, v) is the face's authored texture coordinate (0 when the cage
// carries no UV). When set and slice_scene is given an out_segmentation, the scene is classified per refined
// face and split into per-channel layer contours. Empty = single colour.
struct ColorField {
    std::function<V3(const V3 &point, const V3 &normal, const V3 &dPdx, const V3 &dPdy, double u, double v)> eval;
    std::vector<FlushPredict::RGBColor> palette;
    double band_width  = 1.0;    // mm: width of the colour ribbon deposited along a wall (>= a few perimeters)
    double wall_depth  = 0.0;    // mm: depth of the outer-wall shell the colour claims; 0 -> legacy thin-ribbon clip
    bool   dither      = false;  // spatially dither Cout across the two nearest filaments, else hard quantize
    double dither_cell = 0.5;    // mm: dither pattern cell (~ a line width); ignored unless `dither`
    explicit operator bool() const { return bool(eval) && ! palette.empty(); }
};

// Slice an instanced scene into per-layer contours by amplifying one prototype through each
// placement, unioned -- the N-instance geometry is never baked. Mirrors slice_mesh_ex so it
// stands in at the slice_volume seam. Divergences from slicing one combined mesh: vase mode
// keeps N largest contours, and closing_radius closes each instance before the union.
// out_segmentation (when non-null AND a ColorField is given): filled with [layer][channel] contours,
// channel k = color.palette[k]. Each channel gets the wall ribbons of its colour clipped to the layer
// (a within-layer pattern); the merged per-layer contours are returned as usual, unaffected by colour.
// `displacement` is material 0 (the whole surface, or the fallback for any face in no subset);
// `extra_displacements[k]` is material k+1, selected per control face by SubdivCage::face_material.
// Empty -> single material, the pre-subset path unchanged. Colour is still whole-surface (material 0).
std::vector<ExPolygons> slice_scene(
    const PrintManScene         &scene,
    const MeshSlicingParamsEx   &params,
    const std::vector<float>    &zs,
    const std::function<void()> &throw_on_cancel = [](){},
    const std::function<void(size_t, size_t)> &report_progress = {},
    const DisplacementField     &displacement = {},
    const ColorField            &color = {},
    std::vector<std::vector<ExPolygons>> *out_segmentation = nullptr,
    const std::vector<DisplacementField> &extra_displacements = {});

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_Engine_hpp_
