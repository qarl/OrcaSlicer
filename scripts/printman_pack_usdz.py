#!/usr/bin/env python3
"""Bundle a PrintMan USD and its OSL shaders + texture maps into one self-contained .usdz.

A PrintMan material binds OSL shaders by info:id; the compiled shaders (.oso) and the texture maps (.tx)
they sample normally sit on a build-time searchpath, so a model is not one portable file. This packs the
USD's bound shaders and the maps you name into the .usdz alongside the geometry, so it slices anywhere --
OrcaSlicer's loader extracts the bundle at load time (see USD.cpp extract_usdz_shader_assets).

    printman_pack_usdz.py INPUT.usd OUTPUT.usdz [--shaders DIR] [MAP ...]

--shaders DIR   where the compiled .oso and .tx live (default: build/arm64/osl_shaders under the repo).
MAP ...         texture maps to bundle (e.g. earth.tx earth_height.tx); resolved against --shaders if
                not an absolute/existing path. The bound surface/displacement shaders' .oso are added
                automatically from --shaders.

Example -- the earth globe demo (surface printman_earth + displacement printman_earth_disp, both maps):

    printman_pack_usdz.py tests/data/test_usd/earth_globe.usda earth_globe.usdz \\
        --shaders build/arm64/osl_shaders earth.tx earth_height.tx
"""
import argparse, os, sys
from pxr import Usd, UsdShade, Sdf


def bound_shader_ids(stage):
    ids = []
    for prim in stage.Traverse():
        rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        mat = rel[0] if isinstance(rel, tuple) else rel
        if not mat:
            continue
        for src in (mat.ComputeSurfaceSource(), mat.ComputeDisplacementSource()):
            sh = src[0] if isinstance(src, tuple) else src
            if not sh:
                continue
            r = sh.GetShaderId()
            sid = r[1] if isinstance(r, tuple) else r
            if sid and sid not in ids:
                ids.append(sid)
    return ids


def main():
    ap = argparse.ArgumentParser(description="Pack a PrintMan USD + its shaders/maps into a self-contained .usdz")
    ap.add_argument("input");  ap.add_argument("output")
    ap.add_argument("--shaders", default=None, help="dir holding the .oso/.tx (default: build/arm64/osl_shaders)")
    ap.add_argument("maps", nargs="*", help="texture maps to bundle (resolved against --shaders)")
    a = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shaders = a.shaders or os.path.join(repo, "build/arm64/osl_shaders")

    stage = Usd.Stage.Open(a.input)
    if not stage:
        sys.exit("cannot open " + a.input)
    assets = []
    for sid in bound_shader_ids(stage):
        oso = os.path.join(shaders, sid + ".oso")
        if not os.path.exists(oso):
            sys.exit("missing compiled shader " + oso + " (build first, or pass --shaders)")
        assets.append(oso)
    for m in a.maps:
        p = m if os.path.exists(m) else os.path.join(shaders, m)
        if not os.path.exists(p):
            sys.exit("missing map " + m)
        assets.append(p)

    layer = os.path.splitext(os.path.basename(a.output))[0] + ".usdc"
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.output)) or ".", "." + layer)
    stage.GetRootLayer().Export(tmp)                       # binary crate layer (compact)
    if os.path.exists(a.output):
        os.remove(a.output)
    w = Sdf.ZipFileWriter.CreateNew(a.output)
    w.AddFile(tmp, layer)                                  # first file = default layer
    for p in assets:
        w.AddFile(p, os.path.basename(p))
    w.Save()
    os.remove(tmp)
    print("wrote %s (%.1f MB): %s" % (a.output, os.path.getsize(a.output) / 1e6,
                                      ", ".join([layer] + [os.path.basename(p) for p in assets])))


if __name__ == "__main__":
    main()
