#pragma once
//
// USD front-end (declaration). The core is not USD-bound; this edge module turns a USD stage into
// a neutral cage the amplifier consumes. Implemented in src/usd/usd_read.cpp, compiled only under
// PRINTMAN_USD (pxr is a fenced heavy dep).
//
#include <array>
#include <string>
#include <vector>

namespace printman {

// One resolved material. A mesh can bind several through UsdGeomSubsets (materialBind family), so a
// cage carries a list of these plus a per-face index into it.
struct UsdMaterial {
    std::string osl_disp;       // info:id of the displacement shader
    std::string osl_color;      // info:id of the surface shader
    double max_magnitude = 0;   // inputs:printman:maxMagnitude
    bool object_space = false;  // inputs:printman:objectSpace on the surface shader -- colour sees
                                // object-normalized Z (0 at the bottom layer, 1 at the top)
    // USD convention fallback for a material we cannot evaluate: primvars:displayColor, else a
    // UsdPreviewSurface constant diffuseColor, else the 0.18 neutral gray. Zero displacement.
    std::array<float, 3> fallback_color{{0.18f, 0.18f, 0.18f}};
    // When UsdPreviewSurface's diffuseColor is texture-connected (UsdUVTexture), the encoded image
    // bytes (resolved out of the .usdz / off disk via the asset resolver), for sampling as albedo.
    std::vector<unsigned char> diffuse_tex;
    std::string diffuse_tex_name;          // asset basename -- gives OIIO the format (extension)
    bool diffuse_tex_srgb = true;          // decode sRGB->linear on read (false if sourceColorSpace=raw)
};

struct UsdCage {
    std::vector<std::array<double, 3>> points;
    std::vector<int> counts;    // per-face vertex count
    std::vector<int> indices;   // face-vertex indices (CSR w/ counts)
    std::vector<std::array<double, 2>> st;  // face-varying UV, parallel to indices (empty if none)
    std::string subdiv_scheme = "catmullClark";
    std::string up_axis = "Z";   // stage upAxis as authored; points below are already Z-up normalized
    // Materials bound to this mesh (index 0 is the mesh-level default), and a per-face index into
    // them (empty -> every face uses material 0). Subsets refine per-face; index 0 covers the rest.
    std::vector<UsdMaterial> materials;
    std::vector<int> face_material;         // size counts.size() when subsets exist, else empty
    // Semi-sharp subdivision tags (Catmull-Clark). crease_indices are vertex-index chains grouped by
    // crease_lengths; crease_sharpnesses is one per chain or one per edge. corner_indices/sharpnesses
    // tag sharp vertices. boundary is UsdGeom interpolateBoundary (0 none / 1 edgeOnly / 2 edgeAndCorner
    // -- matches subdiv::BOUNDARY_*). triangle_smooth is triangleSubdivisionRule == "smooth". The
    // defaults below (empty tags, edgeAndCorner, non-smooth) reproduce the old hard-coded build_cage
    // args exactly, so a smooth cage is unchanged.
    std::vector<int> crease_indices, crease_lengths;
    std::vector<double> crease_sharpnesses;
    std::vector<int> corner_indices;
    std::vector<double> corner_sharpnesses;
    int boundary = 2;              // subdiv::BOUNDARY_EDGE_AND_CORNER
    bool triangle_smooth = false;  // triangleSubdivisionRule == "smooth"
};

// Load every renderable Mesh prim (each with its own bound material) from a USD file, as a vector
// of cages. Each cage's points are already baked to world space and normalized to Z-up. Skipped
// counts proxy/guide/invisible prims that were dropped. plugin_dir is the OpenUSD plugInfo tree to
// register. Returns false with err set on failure (including a stage with no printable mesh).
bool load_usd_scene(const std::string& path, const std::string& plugin_dir,
                    std::vector<UsdCage>& out, int& skipped, std::string& err);

}  // namespace printman
