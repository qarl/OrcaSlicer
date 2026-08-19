#pragma once
//
// Neutral geometry vocabulary. The core exposes ONLY these types across its API — never a host
// type (libslic3r ExPolygon / Cura Polygons / Eigen). This is the portability mechanism: a host
// converts its own types to/from these at the seam. Positions are doubles in millimetres.
//
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace printman {

using V2 = std::array<double, 2>;
using V3 = std::array<double, 3>;

// A closed loop of 2D points; winding sign carries orientation (non-zero fill at slice time).
using Ring = std::vector<V2>;

// A solid region: one outer ring plus holes.
struct Region {
    Ring outer;
    std::vector<Ring> holes;
};

// One printed layer: the solid cross-section at height z. Colour stays albedo RGB in the core
// (filament/tank quantization is a per-host/per-sink step), so colour tagging is added with
// shading rather than baked to a filament index here.
struct LayerContours {
    double z = 0.0;
    std::vector<Region> regions;
};

// The core's neutral product: an ordered stack of per-layer contours.
using ContourStack = std::vector<LayerContours>;

// Transient per-band mesh: indexed float triangles + per-vertex attributes. Float positions so
// shared-corner bit-identity (watertightness) survives. NEVER returned whole — one band lives at
// a time (the REYES memory bound).
struct Mesh {
    std::vector<std::array<float, 3>> pos;
    std::vector<std::array<float, 3>> nrm;
    std::vector<std::array<float, 2>> uv;
    // Arbitrary Output Variables: the shader declares named channels (colour "Cout", normals,
    // displacement, custom masks, ...). aov_names[k] names channel k; aov[k][vid] is its vec3
    // value at vertex vid. The core carries them all; the PNG/voxel sink selects which to emit.
    std::vector<std::string> aov_names;
    std::vector<std::vector<std::array<float, 3>>> aov;
    std::vector<std::array<std::uint32_t, 3>> tri;

    int aov_index(const std::string& name) const {
        for (std::size_t k = 0; k < aov_names.size(); ++k)
            if (aov_names[k] == name) return int(k);
        return -1;
    }
};

}  // namespace printman
