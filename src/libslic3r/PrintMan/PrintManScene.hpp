#ifndef slic3r_PrintMan_PrintManScene_hpp_
#define slic3r_PrintMan_PrintManScene_hpp_

#include <array>
#include <map>
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

    // Optional displacement shader, read from the USD prim (printman:oslShader / printman:maxDisplacement).
    // The name is a compiled .oso on the OSL shader searchpath; the bound grows the slice band so the
    // relief is not clipped. Only honoured when the build links OSL (SLIC3R_OSL); empty = no displacement.
    std::string osl_shader;
    double      osl_max_displacement = 0.0;
};

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_PrintManScene_hpp_
