#!/usr/bin/env python3
"""Generate a tall subdivision prism (.usda) that refines to MANY Z-bands.

A square-section prism, tessellated into stacked rings so each face spans only a small Z
range -> the slice engine processes it in many bands, exercising cross-band seam continuity
(the single-face cube refines to one band and hides seams)."""

N = 10          # levels along Z
H = 40.0        # height mm
S = 2.0         # half-width mm (4x4 cross section)

verts = []
for i in range(N + 1):
    z = H * i / N
    verts += [(-S, -S, z), (S, -S, z), (S, S, z), (-S, S, z)]

faces = []  # each is a 4-tuple of vertex indices
faces.append((0, 3, 2, 1))                       # bottom (normal -Z)
for i in range(N):
    b, t = 4 * i, 4 * (i + 1)
    for j in range(4):
        k = (j + 1) % 4
        faces.append((b + j, b + k, t + k, t + j))  # side quad, outward
faces.append((4 * N + 0, 4 * N + 1, 4 * N + 2, 4 * N + 3))  # top (normal +Z)

fvc = ", ".join("4" for _ in faces)
fvi = ", ".join(str(idx) for f in faces for idx in f)
pts = ", ".join(f"({v[0]}, {v[1]}, {v[2]})" for v in verts)

print(f"""#usda 1.0
(
    metersPerUnit = 0.001
    upAxis = "Z"
)

def Mesh "prism"
{{
    int[] faceVertexCounts = [{fvc}]
    int[] faceVertexIndices = [{fvi}]
    point3f[] points = [{pts}]
    uniform token subdivisionScheme = "catmullClark"
}}""")
