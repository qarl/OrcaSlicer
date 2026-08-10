#ifndef slic3r_PrintMan_PrintManScene_hpp_
#define slic3r_PrintMan_PrintManScene_hpp_

#include <vector>

#include "libslic3r/Point.hpp"        // Transform3d
#include "libslic3r/TriangleMesh.hpp" // indexed_triangle_set

namespace Slic3r { namespace PrintMan {

// One placement: which prototype, and its transform (column-vector mm frame, P = xform * v).
struct Placement
{
    int         prototype = 0;
    Transform3d xform     = Transform3d::Identity();
};

// A PrintMan volume's parametric geometry: prototype cages and the placements that instance them.
struct PrintManScene
{
    std::vector<indexed_triangle_set> prototypes;
    std::vector<Placement>            placements;
};

}} // namespace Slic3r::PrintMan

#endif // slic3r_PrintMan_PrintManScene_hpp_
