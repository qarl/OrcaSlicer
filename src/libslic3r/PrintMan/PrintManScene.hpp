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

// One authored OSL shader parameter carried from the USD material -- an `inputs:<name>` on the shader prim,
// outside the printman: namespace (which is host directives, not shader params). Forwarded to the OSL shader
// at load so a material can tune its shader (e.g. a grid's Pitch, or a colour) instead of running on the .osl
// defaults. The type is kept so the value reaches OSL with the matching TypeDesc: a colour must be set as a
// colour, not a bare float triple, or OSL rejects it. Covers the scalar, 3-float, and string types.
struct ShaderParam
{
    enum class Type { Float, Int, Color, Vector, Point, Normal, String };
    std::string name;                       // the OSL parameter name, e.g. "Pitch"
    Type        type = Type::Float;
    double      x = 0.0, y = 0.0, z = 0.0;   // Float/Int in x; Color/Vector/Point/Normal in x, y, z
    std::string str;                         // String
};

// The OSL shaders of one material -- the same shape USD models shading (a `surface` and a `displacement`
// terminal, each an .oso on the searchpath) plus the PrintMan colour directives. A prim binds one material
// to the whole mesh (its default), and a UsdGeomSubset may bind another to a face region; PrintManScene
// carries the mesh material as its scalar osl_*/color_* fields (material 0) and any region materials in
// extra_materials, with SubdivCage::face_material selecting per control face. See read_printman_material.
struct MaterialShaders
{
    std::string surface;                 // material:surface      -> Cout (colour)
    std::string displacement;            // material:displacement -> Disp (relief)
    double      max_displacement = 0.0;  // printman:maxMagnitude -- grows the slice band so relief is not clipped
    bool        object_space  = false;   // printman:objectSpace  -- feed the colour shader an object-normalized Z
    bool        dither        = false;   // printman:dither       -- spatially blend across the two nearest filaments
    bool        all_filaments = false;   // printman:allFilaments -- dither across every loaded filament
    bool        km            = false;   // printman:colorKM      -- classify via the Kubelka-Munk solver, not the RGB dither
    std::vector<ShaderParam> surface_params;       // authored inputs on the surface shader prim
    std::vector<ShaderParam> displacement_params;  // authored inputs on the displacement shader prim
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
    // Per control face, which material paints + displaces it: 0 = the scene's default material (its scalar
    // osl_*/color_* fields), k>=1 = scene.extra_materials[k-1], authored via UsdGeomSubset face bindings.
    // Empty = every face is material 0 (the single-material case), so the index maps straight to the core's
    // per-control-face material tag (ctag). Length, when set, is the control-face count (face_counts.size()).
    std::vector<int>                   face_material;
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
    // Authored parameters on material 0's shaders, forwarded to OSL at load so the material can tune its
    // shader (e.g. a grid's Pitch) instead of running on the .osl defaults. See PrintObjectSlice.
    std::vector<ShaderParam> osl_surface_params;
    std::vector<ShaderParam> osl_displacement_params;
    // Region materials beyond the default: a UsdGeomSubset may bind a different material to a face region,
    // so a single mesh prints each region in its own surface + displacement shader. The scalar fields above
    // are material 0 (the mesh's own binding, and the fallback for any face in no subset); extra_materials[k]
    // is material k+1, selected per control face by SubdivCage::face_material. Empty = single-material (the
    // scalar fields alone), which leaves every existing slice path byte-identical.
    std::vector<MaterialShaders> extra_materials;
    // Where this scene's compiled shaders (.oso) and texture maps (.tx) live. Empty = the built-in
    // PRINTMAN_OSL_SHADER_DIR. A self-contained .usdz bundles them, so the loader extracts them once to a
    // cache dir and records it here, and the whole model -- geometry, shaders, maps -- is one portable file.
    std::string osl_shader_searchpath;

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

    // When set (printman:colorKM), the surface colour is classified with the vendored FullSpectrum
    // Kubelka-Munk solver -- a real filament pigment mix -- instead of the built-in linear-RGB dither.
    bool color_km = false;
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
