#ifndef slic3r_PrintMan_PrintManScene_hpp_
#define slic3r_PrintMan_PrintManScene_hpp_

#include <array>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "libslic3r/Point.hpp"        // Transform3d
#include "libslic3r/TriangleMesh.hpp" // indexed_triangle_set

namespace Slic3r { namespace PrintMan {

// One placement of a prototype (P = xform * v, mm).
struct Placement
{
    int         prototype = 0;
    Transform3d xform     = Transform3d::Identity();
};

// A subdivision control cage as raw USD attributes in the prototype's local frame, refined at slice
// time (see Format/USDSubdiv.hpp). The paired prototype mesh is its coarse control mesh.
struct SubdivCage
{
    std::vector<std::array<double, 3>> points;          // control points, prototype-local
    std::vector<int>                   face_counts;     // vertices per face
    std::vector<int>                   face_indices;    // concatenated face-vertex indices
    std::vector<int>                   crease_indices;  // USD creaseIndices / Lengths / Sharpnesses
    std::vector<int>                   crease_lengths;
    std::vector<double>                crease_sharp;
    std::vector<int>                   corner_indices;  // USD cornerIndices / Sharpnesses
    std::vector<double>                corner_sharp;
    std::vector<std::array<double, 2>> st;              // primvars:st, one (u,v) per face-corner; empty = none
    int                                boundary        = 2;      // usd_subdiv::Boundary
    bool                               triangle_smooth = false;  // triangleSubdivisionRule == "smooth"
    bool                               flip_winding    = false;  // orientation == leftHanded
    std::string                        scheme;                   // catmullClark | loop | bilinear
    // AABB of the refined surface (not the control cage, which over-estimates), for bed placement.
    std::array<double, 3>              refined_lo{{0, 0, 0}};
    std::array<double, 3>              refined_hi{{0, 0, 0}};
};

// A PrintMan volume's parametric geometry: prototype cages and the placements that instance them.
struct PrintManScene
{
    std::vector<indexed_triangle_set> prototypes;
    std::vector<Placement>            placements;
    std::map<int, SubdivCage>         cages;   // prototype index -> cage; absent = a plain mesh

    // OSL shaders resolved from the prim's bound UsdShade material -- one per material terminal, exactly as
    // USD models shading: `surface` drives the per-face colour (Cout), `displacement` drives the relief (Disp).
    // Each name is a compiled .oso on the OSL shader searchpath (the shader's info:id). One shader may back
    // both terminals (the UsdPreviewSurface pattern); either may be empty. max_displacement grows the slice
    // band so the relief is not clipped. Only honoured when the build links OSL (SLIC3R_OSL).
    std::string osl_surface_shader;         // material:surface      -> Cout (colour)
    std::string osl_displacement_shader;    // material:displacement -> Disp (relief)
    double      osl_max_displacement = 0.0;

    // Colour: the 1-based filament ids this scene paints with. Empty = single-filament (no colour).
    // When set, slice_scene also emits per-filament contours and the object is split into one print
    // region per filament, reusing Orca's MMU region machinery (see apply_printman_mm_segmentation).
    std::vector<unsigned int> filaments;

    // Colour-shader hints, read as inputs on the surface shader (inputs:objectSpace / inputs:dither).
    // object_space feeds the colour shader an object-normalized Z (0 at the bottom layer, 1 at the top) so a gradient
    // spans the whole object at any height; dither spatially blends a continuous colour across the two
    // nearest filaments (else the colour is hard-quantized to the single nearest). Both default off, so a
    // categorical shader (the grid) keeps world coords and crisp filament regions.
    bool color_object_space = false;
    bool color_dither       = false;

    // When set, the scene paints with EVERY loaded filament -- the surface colour is dithered across the
    // whole loaded palette (each face's two nearest), not just the explicit `filaments` list. The concrete
    // ids depend on how many filaments are loaded, so they are resolved at apply/slice time (see
    // resolved_filaments), not stored here. Off -> `filaments` above is the literal set.
    bool color_all_filaments = false;
};

// The concrete 1-based filament ids this scene paints with: the explicit `filaments` list, or -- when
// color_all_filaments is set -- every loaded filament {1..n_loaded}, so the surface colour dithers across
// the whole loaded palette. n_loaded is the number of filaments loaded in the print config; the returned
// ids are in [1, n_loaded] by construction.
inline std::vector<unsigned int> resolved_filaments(const PrintManScene &s, unsigned int n_loaded)
{
    if (s.color_all_filaments) {
        std::vector<unsigned int> v(n_loaded);
        std::iota(v.begin(), v.end(), 1u);
        return v;
    }
    return s.filaments;
}

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_PrintManScene_hpp_
