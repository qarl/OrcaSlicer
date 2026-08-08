#!/usr/bin/env python3
"""Generate a spherical SHELL of little instances (torus or sphere): ONE prototype,
instanced N times with centers on a big sphere of radius R = ratio*r via a Fibonacci
(golden-angle) distribution. For a torus, each instance is ORIENTED so its axis (local
+Z, the axis through the hole) aligns with the big sphere's surface normal at that point
-- little rings lying flat on the surface, holes facing outward.

The prototype is authored ONCE: amplified holds it + N placements; naive materializes N
copies. Placements are written as full 4x4 matrices (matrix4d xformOp:transform) so the
per-instance rotation composes through the reader's GetLocalToWorldTransform.

Usage: gen_shell.py <out.usda> --n N --ratio R/r --r <mm> --tube <mm> --shape torus|sphere
"""
import argparse
import math


def sphere(u_segs, v_segs, radius, minor=None):
    # NOTE: unlike torus() this leaves the longitude seam duplicated and fans each pole
    # (non-manifold). A sphere's horizontal slice is a single convex loop the slicer closes
    # cleanly, so it hasn't shown the torus's sliver pathology; weld it like torus() if it does.
    pts = []
    for i in range(u_segs + 1):
        theta = math.pi * i / u_segs
        z = radius * math.cos(theta)
        rxy = radius * math.sin(theta)
        for j in range(v_segs + 1):
            phi = 2.0 * math.pi * j / v_segs
            pts.append((rxy * math.cos(phi), rxy * math.sin(phi), z))
    return pts, _grid_tris(u_segs, v_segs)


def torus(u_segs, v_segs, major, minor):
    # Ring in local XY, axis = local +Z (through the hole); u around the ring, v around the
    # tube. WELDED: exactly u_segs x v_segs unique vertices with the seam closed by modulo
    # wrapping, so seam vertices are SHARED. A duplicated-seam torus is topologically open and
    # the slicer shatters it into non-deterministic slivers (on the naive flatten too).
    pts = []
    for i in range(u_segs):
        u = 2.0 * math.pi * i / u_segs
        cu, su = math.cos(u), math.sin(u)
        for j in range(v_segs):
            v = 2.0 * math.pi * j / v_segs
            rr = major + minor * math.cos(v)
            pts.append((rr * cu, rr * su, minor * math.sin(v)))
    return pts, _wrap_tris(u_segs, v_segs)


def _grid_tris(u_segs, v_segs):
    tris = []
    row = v_segs + 1
    for i in range(u_segs):
        for j in range(v_segs):
            a = i * row + j
            b = (i + 1) * row + j
            c = (i + 1) * row + (j + 1)
            d = i * row + (j + 1)
            tris.append((a, b, c))
            tris.append((a, c, d))
    return tris


def _wrap_tris(u_segs, v_segs):
    # Grid triangles over a u_segs x v_segs vertex lattice with BOTH directions wrapped
    # (indices modulo), i.e. a closed manifold whose seam vertices are shared, not duplicated.
    tris = []
    for i in range(u_segs):
        for j in range(v_segs):
            a = i * v_segs + j
            b = ((i + 1) % u_segs) * v_segs + j
            c = ((i + 1) % u_segs) * v_segs + (j + 1) % v_segs
            d = i * v_segs + (j + 1) % v_segs
            tris.append((a, b, c))
            tris.append((a, c, d))
    return tris


def fibonacci_shell(n, R):
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        rxy = math.sqrt(max(0.0, 1.0 - z * z))
        th = ga * i
        out.append((R * rxy * math.cos(th), R * rxy * math.sin(th), R * z))
    return out


def _norm(v):
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def placement_matrix(center):
    # Rotation mapping local +Z -> surface normal n (= center direction), with an arbitrary
    # but stable tangent basis (ex, ey). USD is row-vector (world = local . M), so the first
    # three matrix ROWS are the world images of local x, y, z, and row 3 is the translation.
    n = _norm(center)
    a = (0.0, 0.0, 1.0) if abs(n[2]) < 0.99 else (1.0, 0.0, 0.0)
    ex = _norm(_cross(a, n))
    ey = _cross(n, ex)  # already unit
    return (ex, ey, n, center)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--ratio", type=float, default=30.0)   # R_shell / r
    ap.add_argument("--r", type=float, default=1.0)        # sphere radius / torus MAJOR radius, mm
    ap.add_argument("--tube", type=float, default=0.35)    # torus MINOR (tube) radius, mm
    ap.add_argument("--shape", choices=["torus", "sphere"], default="torus")
    ap.add_argument("--stacks", type=int, default=90)      # u segments
    ap.add_argument("--slices", type=int, default=90)      # v segments
    args = ap.parse_args()

    R = args.ratio * args.r
    if args.shape == "torus":
        pts, tris = torus(args.stacks, args.slices, args.r, args.tube)
    else:
        pts, tris = sphere(args.stacks, args.slices, args.r)
    centers = fibonacci_shell(args.n, R)

    points_str = ", ".join("(%g, %g, %g)" % p for p in pts)
    counts_str = ", ".join("3" for _ in tris)
    idx_str = ", ".join("%d, %d, %d" % t for t in tris)

    with open(args.out, "w") as f:
        f.write('#usda 1.0\n(\n    metersPerUnit = 0.001\n    upAxis = "Z"\n)\n\n')
        f.write('def "Prototypes"\n{\n    class Xform "Proto"\n    {\n')
        f.write('        def Mesh "Inst"\n        {\n')
        f.write('            int[] faceVertexCounts = [%s]\n' % counts_str)
        f.write('            int[] faceVertexIndices = [%s]\n' % idx_str)
        f.write('            point3f[] points = [%s]\n' % points_str)
        f.write('            uniform token subdivisionScheme = "none"\n')
        f.write('        }\n    }\n}\n\n')
        f.write('def "Root"\n{\n')
        for k, c in enumerate(centers):
            ex, ey, n, t = placement_matrix(c)
            f.write('    def Xform "inst_%05d" (\n' % k)
            f.write('        prepend inherits = </Prototypes/Proto>\n')
            f.write('        instanceable = true\n    )\n    {\n')
            f.write('        matrix4d xformOp:transform = ( (%g, %g, %g, 0), (%g, %g, %g, 0), (%g, %g, %g, 0), (%g, %g, %g, 1) )\n'
                    % (ex[0], ex[1], ex[2], ey[0], ey[1], ey[2], n[0], n[1], n[2], t[0], t[1], t[2]))
            f.write('        uniform token[] xformOpOrder = ["xformOp:transform"]\n')
            f.write('    }\n\n')
        f.write('}\n')

    kind = ("torus major=%g tube=%g" % (args.r, args.tube)) if args.shape == "torus" else ("sphere r=%g" % args.r)
    print("wrote %s: %s, %d tris/%d verts, %d instances on shell R=%g (ratio %g), normal-aligned"
          % (args.out, kind, len(tris), len(pts), args.n, R, args.ratio))


if __name__ == "__main__":
    main()
