#pragma once
// Catmull-Clark / Loop / bilinear subdivision, carrying face-varying UV. Copied verbatim from the
// fork's std-only, OpenSubdiv-validated USDSubdiv.hpp (namespace renamed). Zero external deps.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace printman { namespace subdiv {

using Pt = std::array<double, 3>;

inline Pt operator+(const Pt &a, const Pt &b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Pt operator*(const Pt &a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline void add_to(Pt &a, const Pt &b) { a[0] += b[0]; a[1] += b[1]; a[2] += b[2]; }

static const double SHARPNESS_INFINITE = 10.0;

// Device-perfect slice level is clamped to this range. The proxy AABB is refined at kDeviceLevelMax
// (the finest, most contracted slice), so a coarser actual slice is never smaller and never floats.
static const int kDeviceLevelMin = 2;
static const int kDeviceLevelMax = 8;

// Boundary interpolation, matching Sdc::Options::VtxBoundaryInterpolation.
enum Boundary { BOUNDARY_NONE = 0, BOUNDARY_EDGE_ONLY = 1, BOUNDARY_EDGE_AND_CORNER = 2 };

struct Cage {
    std::vector<Pt> verts;
    std::vector<int> fvi, foff;               // faces in CSR form
    std::map<long long, double> edge_crease;  // undirected-edge key -> sharpness
    std::vector<double> corner_sharp;         // per vertex, 0 where untagged
    // Optional face-varying UV (primvars:st), one (u,v) per face-corner -- parallel to fvi, so a seam
    // vertex carries a distinct UV on each side (what vertex interpolation cannot do). Refined bilinearly
    // (per face, no cross-face averaging), matching USD's faceVaryingLinearInterpolation = "all". Empty =
    // the cage carries no UV. Only Catmull-Clark refinement (catmull_clark/refine_region) propagates it.
    std::vector<std::array<double, 2>> fvar;
    int boundary = BOUNDARY_EDGE_AND_CORNER;
    bool triangle_smooth = false;             // triangleSubdivisionRule == "smooth"
    int nfaces() const { return int(foff.size()) - 1; }
    int nverts() const { return int(verts.size()); }
    bool has_uv() const { return fvar.size() == fvi.size() && ! fvi.empty(); }
};

inline long long ekey(int a, int b, long long nv) {
    return (long long)std::min(a, b) * nv + std::max(a, b);
}
inline double decrement(double s) { return s >= SHARPNESS_INFINITE ? SHARPNESS_INFINITE : (s > 1.0 ? s - 1.0 : 0.0); }

struct Topo {
    std::vector<int> corner_vert, corner_face, next_edge, prev_edge;
    std::vector<std::pair<int, int>> edge_verts;
    int ne = 0;
};

inline Topo build_topo(const Cage &c) {
    Topo t;
    int nv = c.nverts(), nf = c.nfaces(), C = int(c.fvi.size());
    t.corner_vert.resize(C); t.corner_face.resize(C);
    t.next_edge.resize(C); t.prev_edge.resize(C);
    // Hash map, not std::map: it only dedups edges (ids are encounter-ordered), so the output is
    // identical, but the tree's per-lookup cost dominated each step. reserve avoids rehashing.
    std::unordered_map<long long, int> id;
    id.reserve(size_t(C));
    for (int f = 0; f < nf; ++f) {
        int s = c.foff[f], e = c.foff[f + 1];
        for (int k = s; k < e; ++k) {
            t.corner_vert[k] = c.fvi[k];
            t.corner_face[k] = f;
            int a = c.fvi[k], b = c.fvi[(k + 1 == e) ? s : k + 1];
            long long key = ekey(a, b, nv);
            auto it = id.find(key);
            int eid;
            if (it == id.end()) {
                eid = int(t.edge_verts.size());
                id[key] = eid;
                t.edge_verts.push_back({std::min(a, b), std::max(a, b)});
            } else eid = it->second;
            t.next_edge[k] = eid;
        }
        for (int k = s; k < e; ++k) t.prev_edge[k] = t.next_edge[(k == s) ? e - 1 : k - 1];
    }
    t.ne = int(t.edge_verts.size());
    return t;
}

inline std::vector<double> read_crease(const Cage &c, const Topo &t) {
    std::vector<double> s(t.ne, 0.0);
    long long nv = c.nverts();
    for (int e = 0; e < t.ne; ++e) {
        auto it = c.edge_crease.find(ekey(t.edge_verts[e].first, t.edge_verts[e].second, nv));
        if (it != c.edge_crease.end()) s[e] = it->second;
    }
    return s;
}

inline void propagate(Cage &out, int nv, int face_block, const Topo &t,
                      const std::vector<double> &crease, const std::vector<double> &corner) {
    long long onv = out.nverts();
    for (int e = 0; e < t.ne; ++e) {
        double sc = decrement(crease[e]);
        if (sc > 0.0) {
            int a = t.edge_verts[e].first, b = t.edge_verts[e].second, ep = nv + face_block + e;
            out.edge_crease[ekey(a, ep, onv)] = sc;
            out.edge_crease[ekey(ep, b, onv)] = sc;
        }
    }
    out.corner_sharp.assign(out.nverts(), 0.0);
    for (int v = 0; v < nv; ++v) {
        double sc = decrement(corner[v]);
        if (sc > 0.0) out.corner_sharp[v] = sc;
    }
}

inline std::vector<Pt> rule_pos(const Cage &c, const Topo &t, const std::vector<double> &s_edge,
                                const std::vector<double> &cflag, double thr, const std::vector<Pt> &p_smooth) {
    int nv = c.nverts();
    std::vector<int> count(nv, 0);
    std::vector<Pt> nbr(nv, Pt{0, 0, 0});
    for (int e = 0; e < t.ne; ++e)
        if (s_edge[e] > thr) {
            int a = t.edge_verts[e].first, b = t.edge_verts[e].second;
            count[a]++; count[b]++;
            add_to(nbr[a], c.verts[b]); add_to(nbr[b], c.verts[a]);
        }
    std::vector<Pt> pos(nv);
    for (int v = 0; v < nv; ++v) {
        bool corner = cflag[v] > thr || count[v] > 2;
        bool crease = count[v] == 2 && !(cflag[v] > thr);
        pos[v] = corner ? c.verts[v] : (crease ? c.verts[v] * 0.75 + nbr[v] * 0.125 : p_smooth[v]);
    }
    return pos;
}

inline std::vector<Pt> blend_vertices(const Cage &c, const Topo &t, const std::vector<double> &s_edge,
                                      const std::vector<double> &corner_eff, const std::vector<Pt> &p_smooth,
                                      const std::vector<int> &edge_valence) {
    int nv = c.nverts();
    auto p_parent = rule_pos(c, t, s_edge, corner_eff, 0.0, p_smooth);
    auto p_child = rule_pos(c, t, s_edge, corner_eff, 1.0, p_smooth);
    std::vector<double> tsum(nv, 0.0), tcount(nv, 0.0);
    for (int e = 0; e < t.ne; ++e)
        if (s_edge[e] > 0.0 && s_edge[e] <= 1.0) {
            int a = t.edge_verts[e].first, b = t.edge_verts[e].second;
            tsum[a] += s_edge[e]; tsum[b] += s_edge[e];
            tcount[a] += 1; tcount[b] += 1;
        }
    for (int v = 0; v < nv; ++v)
        if (corner_eff[v] > 0.0 && corner_eff[v] <= 1.0) { tsum[v] += corner_eff[v]; tcount[v] += 1; }
    std::vector<Pt> out(nv);
    for (int v = 0; v < nv; ++v) {
        double w = tcount[v] > 0 ? std::min(tsum[v] / tcount[v], 1.0) : 0.0;
        out[v] = (edge_valence[v] == 0) ? c.verts[v] : p_parent[v] * w + p_child[v] * (1.0 - w);
    }
    return out;
}

inline Cage catmull_clark(const Cage &c) {
    int nv = c.nverts(), nf = c.nfaces();
    const auto &V = c.verts;
    Topo t = build_topo(c);
    int C = int(c.fvi.size()), ne = t.ne;

    std::vector<Pt> face_points(nf);
    for (int f = 0; f < nf; ++f) {
        Pt sum{0, 0, 0};
        for (int k = c.foff[f]; k < c.foff[f + 1]; ++k) add_to(sum, V[c.fvi[k]]);
        face_points[f] = sum * (1.0 / (c.foff[f + 1] - c.foff[f]));
    }

    std::vector<double> crease = read_crease(c, t), s_edge(ne);
    std::vector<Pt> edge_mid(ne), edge_face_sum(ne, Pt{0, 0, 0});
    std::vector<int> edge_face_count(ne, 0), tri_incident(ne, 0);
    for (int e = 0; e < ne; ++e)
        edge_mid[e] = (V[t.edge_verts[e].first] + V[t.edge_verts[e].second]) * 0.5;
    for (int f = 0; f < nf; ++f) {
        bool tri = (c.foff[f + 1] - c.foff[f]) == 3;
        for (int k = c.foff[f]; k < c.foff[f + 1]; ++k) {
            int e = t.next_edge[k];
            add_to(edge_face_sum[e], face_points[f]);
            edge_face_count[e]++;
            if (tri) tri_incident[e]++;
        }
    }
    for (int e = 0; e < ne; ++e) s_edge[e] = (edge_face_count[e] == 1) ? SHARPNESS_INFINITE : crease[e];

    std::vector<Pt> edge_points(ne);
    for (int e = 0; e < ne; ++e) {
        Pt smooth = (edge_mid[e] * 2.0 + edge_face_sum[e]) * 0.25;
        if (c.triangle_smooth && edge_face_count[e] == 2 && tri_incident[e] >= 1) {
            double fw = 0.5 * (0.470 * tri_incident[e] + 0.25 * (edge_face_count[e] - tri_incident[e]));
            double vw = 0.5 * (1.0 - 2.0 * fw);
            smooth = edge_mid[e] * (2.0 * vw) + edge_face_sum[e] * fw;
        }
        double blend = std::min(std::max(s_edge[e], 0.0), 1.0);
        edge_points[e] = edge_mid[e] * blend + smooth * (1.0 - blend);
    }

    std::vector<Pt> f_sum(nv, Pt{0, 0, 0}), r_sum(nv, Pt{0, 0, 0});
    std::vector<int> face_valence(nv, 0), edge_valence(nv, 0);
    for (int k = 0; k < C; ++k) { add_to(f_sum[t.corner_vert[k]], face_points[t.corner_face[k]]); face_valence[t.corner_vert[k]]++; }
    for (int e = 0; e < ne; ++e) {
        add_to(r_sum[t.edge_verts[e].first], edge_mid[e]);
        add_to(r_sum[t.edge_verts[e].second], edge_mid[e]);
        edge_valence[t.edge_verts[e].first]++; edge_valence[t.edge_verts[e].second]++;
    }
    std::vector<Pt> p_smooth(nv);
    for (int v = 0; v < nv; ++v) {
        double n = edge_valence[v];
        Pt f_avg = f_sum[v] * (1.0 / std::max(face_valence[v], 1));
        Pt r_avg = r_sum[v] * (1.0 / std::max(edge_valence[v], 1));
        p_smooth[v] = (f_avg + r_avg * 2.0 + V[v] * (n - 3.0)) * (1.0 / std::max(n, 1.0));
    }
    std::vector<double> corner_eff = c.corner_sharp;
    if (c.boundary == BOUNDARY_EDGE_AND_CORNER)
        for (int v = 0; v < nv; ++v) if (face_valence[v] == 1) corner_eff[v] = SHARPNESS_INFINITE;

    std::vector<Pt> new_verts = blend_vertices(c, t, s_edge, corner_eff, p_smooth, edge_valence);

    Cage out; out.boundary = c.boundary; out.triangle_smooth = c.triangle_smooth;
    out.verts = new_verts;
    out.verts.insert(out.verts.end(), face_points.begin(), face_points.end());
    out.verts.insert(out.verts.end(), edge_points.begin(), edge_points.end());
    for (int k = 0; k < C; ++k) {
        out.fvi.push_back(t.corner_vert[k]);
        out.fvi.push_back(nv + nf + t.next_edge[k]);
        out.fvi.push_back(nv + t.corner_face[k]);
        out.fvi.push_back(nv + nf + t.prev_edge[k]);
    }
    out.foff.resize(C + 1);
    for (int i = 0; i <= C; ++i) out.foff[i] = 4 * i;
    propagate(out, nv, nf, t, crease, c.corner_sharp);
    // Face-varying UV, refined bilinearly per face (no cross-face averaging, so seams stay split). Each new
    // quad {corner, next-edge, face, prev-edge} matches the fvi push order above, so the pushes stay in sync.
    if (c.has_uv()) {
        out.fvar.reserve(size_t(4) * C);
        for (int f = 0; f < nf; ++f) {
            const int s = c.foff[f], e = c.foff[f + 1];
            std::array<double, 2> favg{0, 0};
            for (int k = s; k < e; ++k) { favg[0] += c.fvar[k][0]; favg[1] += c.fvar[k][1]; }
            const double inv = 1.0 / double(e - s);
            favg[0] *= inv; favg[1] *= inv;
            for (int k = s; k < e; ++k) {
                const int kn = (k + 1 < e) ? k + 1 : s;   // next corner in this face
                const int kp = (k > s) ? k - 1 : e - 1;   // prev corner in this face
                out.fvar.push_back(c.fvar[k]);
                out.fvar.push_back({(c.fvar[k][0] + c.fvar[kn][0]) * 0.5, (c.fvar[k][1] + c.fvar[kn][1]) * 0.5});
                out.fvar.push_back(favg);
                out.fvar.push_back({(c.fvar[k][0] + c.fvar[kp][0]) * 0.5, (c.fvar[k][1] + c.fvar[kp][1]) * 0.5});
            }
        }
    }
    return out;
}

inline Cage loop(const Cage &c) {
    int nv = c.nverts(), nf = c.nfaces();
    const auto &V = c.verts;
    Topo t = build_topo(c);
    int C = int(c.fvi.size()), ne = t.ne;

    std::vector<int> apex(C);
    for (int f = 0; f < nf; ++f) {
        int s = c.foff[f], e = c.foff[f + 1];
        for (int k = s; k < e; ++k) apex[k] = c.fvi[(k == s) ? e - 1 : k - 1];
    }
    std::vector<double> crease = read_crease(c, t), s_edge(ne);
    std::vector<Pt> edge_mid(ne), apex_sum(ne, Pt{0, 0, 0});
    std::vector<int> edge_face_count(ne, 0);
    for (int e = 0; e < ne; ++e)
        edge_mid[e] = (V[t.edge_verts[e].first] + V[t.edge_verts[e].second]) * 0.5;
    for (int k = 0; k < C; ++k) { add_to(apex_sum[t.next_edge[k]], V[apex[k]]); edge_face_count[t.next_edge[k]]++; }
    for (int e = 0; e < ne; ++e) s_edge[e] = (edge_face_count[e] == 1) ? SHARPNESS_INFINITE : crease[e];

    std::vector<Pt> edge_points(ne);
    for (int e = 0; e < ne; ++e) {
        Pt smooth = (edge_face_count[e] == 2) ? edge_mid[e] * 0.75 + apex_sum[e] * 0.125 : edge_mid[e];
        double blend = std::min(std::max(s_edge[e], 0.0), 1.0);
        edge_points[e] = edge_mid[e] * blend + smooth * (1.0 - blend);
    }

    std::vector<Pt> nbr_sum(nv, Pt{0, 0, 0});
    std::vector<int> valence(nv, 0), face_valence(nv, 0);
    for (int e = 0; e < ne; ++e) {
        add_to(nbr_sum[t.edge_verts[e].first], V[t.edge_verts[e].second]);
        add_to(nbr_sum[t.edge_verts[e].second], V[t.edge_verts[e].first]);
        valence[t.edge_verts[e].first]++; valence[t.edge_verts[e].second]++;
    }
    for (int k = 0; k < C; ++k) face_valence[t.corner_vert[k]]++;
    std::vector<Pt> p_smooth(nv);
    for (int v = 0; v < nv; ++v) {
        double n = std::max(valence[v], 1);
        double beta = (0.625 - std::pow(0.375 + 0.25 * std::cos(2.0 * M_PI / n), 2.0)) / n;
        p_smooth[v] = V[v] * (1.0 - valence[v] * beta) + nbr_sum[v] * beta;
    }
    std::vector<double> corner_eff = c.corner_sharp;
    if (c.boundary == BOUNDARY_EDGE_AND_CORNER)
        for (int v = 0; v < nv; ++v) if (face_valence[v] == 1) corner_eff[v] = SHARPNESS_INFINITE;

    std::vector<Pt> new_verts = blend_vertices(c, t, s_edge, corner_eff, p_smooth, valence);

    Cage out; out.boundary = c.boundary; out.triangle_smooth = c.triangle_smooth;
    out.verts = new_verts;
    out.verts.insert(out.verts.end(), edge_points.begin(), edge_points.end());
    for (int k = 0; k < C; ++k) {
        out.fvi.push_back(t.corner_vert[k]);
        out.fvi.push_back(nv + t.next_edge[k]);
        out.fvi.push_back(nv + t.prev_edge[k]);
    }
    for (int f = 0; f < nf; ++f) {
        int s = c.foff[f];
        out.fvi.push_back(nv + t.next_edge[s]);
        out.fvi.push_back(nv + t.next_edge[s + 1]);
        out.fvi.push_back(nv + t.next_edge[s + 2]);
    }
    out.foff.resize(4 * nf + 1);
    for (int i = 0; i <= 4 * nf; ++i) out.foff[i] = 3 * i;
    propagate(out, nv, 0, t, crease, c.corner_sharp);
    return out;
}

inline Cage bilinear(const Cage &c) {
    int nv = c.nverts(), nf = c.nfaces();
    const auto &V = c.verts;
    Topo t = build_topo(c);
    int C = int(c.fvi.size()), ne = t.ne;
    std::vector<Pt> face_points(nf);
    for (int f = 0; f < nf; ++f) {
        Pt sum{0, 0, 0};
        for (int k = c.foff[f]; k < c.foff[f + 1]; ++k) add_to(sum, V[c.fvi[k]]);
        face_points[f] = sum * (1.0 / (c.foff[f + 1] - c.foff[f]));
    }
    std::vector<Pt> edge_points(ne);
    for (int e = 0; e < ne; ++e)
        edge_points[e] = (V[t.edge_verts[e].first] + V[t.edge_verts[e].second]) * 0.5;
    Cage out; out.boundary = c.boundary; out.triangle_smooth = c.triangle_smooth;
    out.verts = V;
    out.verts.insert(out.verts.end(), face_points.begin(), face_points.end());
    out.verts.insert(out.verts.end(), edge_points.begin(), edge_points.end());
    for (int k = 0; k < C; ++k) {
        out.fvi.push_back(t.corner_vert[k]);
        out.fvi.push_back(nv + nf + t.next_edge[k]);
        out.fvi.push_back(nv + t.corner_face[k]);
        out.fvi.push_back(nv + nf + t.prev_edge[k]);
    }
    out.foff.resize(C + 1);
    for (int i = 0; i <= C; ++i) out.foff[i] = 4 * i;
    out.corner_sharp.assign(out.nverts(), 0.0);
    return out;
}

// Build the control cage, expanding USD crease chains (a length-L chain is its L-1 edges;
// creaseSharpnesses is one per chain or one per edge) and corner tags into the cage.
inline Cage build_cage(const std::vector<Pt> &points, const std::vector<int> &counts,
                       const std::vector<int> &indices, const std::vector<int> &creaseIndices,
                       const std::vector<int> &creaseLengths, const std::vector<double> &creaseSharpnesses,
                       const std::vector<int> &cornerIndices, const std::vector<double> &cornerSharpnesses,
                       int boundary, bool triangle_smooth,
                       const std::vector<std::array<double, 2>> &fvar_uv = {}) {
    Cage c;
    c.boundary = boundary;
    c.triangle_smooth = triangle_smooth;
    c.verts = points;
    long long nv = points.size();
    c.foff.push_back(0);
    for (int n : counts) c.foff.push_back(c.foff.back() + n);
    c.fvi = indices;
    c.corner_sharp.assign(points.size(), 0.0);
    // Face-varying UV is one (u,v) per face-corner; ignore a mismatched array rather than desync it.
    if (fvar_uv.size() == indices.size()) c.fvar = fvar_uv;

    long long per_edge_total = 0;
    for (int l : creaseLengths) per_edge_total += std::max(l - 1, 0);
    bool per_edge = (long long)creaseSharpnesses.size() == per_edge_total;
    size_t pos = 0, ei = 0;
    for (size_t ch = 0; ch < creaseLengths.size(); ++ch) {
        int L = creaseLengths[ch];
        // creaseIndices/creaseLengths are independent USD attributes; guard a malformed pair.
        if (L < 0 || pos + size_t(L) > creaseIndices.size()) break;
        for (int j = 0; j + 1 < L; ++j) {
            int a = creaseIndices[pos + j], b = creaseIndices[pos + j + 1];
            double s = per_edge ? (ei < creaseSharpnesses.size() ? creaseSharpnesses[ei] : 0.0)
                                : (ch < creaseSharpnesses.size() ? creaseSharpnesses[ch] : 0.0);
            ++ei;
            if (s > 0.0 && a >= 0 && a < nv && b >= 0 && b < nv) {
                long long k = ekey(a, b, nv);
                auto it = c.edge_crease.find(k);
                c.edge_crease[k] = (it == c.edge_crease.end()) ? s : std::max(it->second, s);
            }
        }
        pos += size_t(L);
    }
    bool has_s = cornerSharpnesses.size() == cornerIndices.size();
    for (size_t i = 0; i < cornerIndices.size(); ++i) {
        int v = cornerIndices[i];
        double s = has_s ? cornerSharpnesses[i] : SHARPNESS_INFINITE;
        if (v >= 0 && v < nv && s > 0.0) c.corner_sharp[v] = std::max(c.corner_sharp[v], s);
    }
    return c;
}

// Refine `levels` times by `scheme` ("catmullClark" / "loop" / "bilinear"); anything else
// (including "none") returns the control cage unchanged.
inline Cage subdivide(Cage c, const std::string &scheme, int levels) {
    for (int i = 0; i < levels; ++i) {
        if (scheme == "catmullClark") c = catmull_clark(c);
        else if (scheme == "loop") c = loop(c);
        else if (scheme == "bilinear") c = bilinear(c);
        else break;
    }
    return c;
}

// Longest control-cage edge (local). The device level resolves this to a sub-tol facet; using the
// longest edge, not the whole-cage diameter, avoids over-refining an already finely tessellated cage.
inline double max_control_edge(const Cage &c) {
    double m = 0.0;
    for (int f = 0; f < c.nfaces(); ++f) {
        int s = c.foff[f], e = c.foff[f + 1];
        for (int k = s; k < e; ++k) {
            const Pt &a = c.verts[c.fvi[k]], &b = c.verts[c.fvi[(k + 1 == e) ? s : k + 1]];
            const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
            m = std::max(m, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    return m;
}

// Largest FINITE authored crease sharpness (0 if none); it floors the level so a semi-sharp crease
// resolves. An infinite crease (s >= SHARPNESS_INFINITE) is excluded -- it stays sharp at every level.
inline double max_crease_sharpness(const Cage &c) {
    double m = 0.0;
    for (const auto &kv : c.edge_crease)
        if (kv.second < SHARPNESS_INFINITE) m = std::max(m, kv.second);
    return m;
}

// Device-perfect level: the longest control edge refines to a sub-`tol` facet in world space (sigma
// scales a local edge to world, tol = min(line width, layer height)), floored so a semi-sharp crease
// resolves, clamped to [2, 8]. The floor of 2 keeps the level-2 proxy AABB a valid outer bound (CC
// contracts inward). `tol == 0` drops the size term; the caller logs when the ceiling is hit.
inline int device_level(double sigma, double max_edge, double tol,
                        double max_crease_sharp, bool *hit_ceiling = nullptr) {
    int L = 2;
    const double world_edge = sigma * max_edge;
    if (tol > 0.0 && world_edge > 0.0)
        L = int(std::ceil(std::log2(std::max(world_edge / tol, 1.0))));
    L = std::max(L, int(std::ceil(std::max(max_crease_sharp, 0.0))));
    L = std::max(L, kDeviceLevelMin);
    if (hit_ceiling) *hit_ceiling = (L > kDeviceLevelMax);   // an over-large object wanted more than the cap
    return std::min(L, kDeviceLevelMax);
}

// Local AABB of the cage's limit surface, for bed placement. Refine toward the limit and stop once it
// converges, so a fine cage need not reach max_level (4^max_level transient faces -- an OOM risk).
// max_level is the finest slice, so this matches the most contracted surface and never floats. The stop
// is floored at floor(s)+1: a semi-sharp feature sits frozen at its control position until it smooths
// there, so stopping earlier records a mis-placed extreme that a finer slice then lifts.
inline void refined_limit_aabb(const Cage &cage, const std::string &scheme, int max_level,
                               Pt &lo, Pt &hi) {
    auto box = [](const Cage &c, Pt &lo, Pt &hi) {
        const double inf = std::numeric_limits<double>::infinity();
        lo = { inf, inf, inf }; hi = { -inf, -inf, -inf };
        for (const Pt &v : c.verts)
            for (int d = 0; d < 3; ++ d) { lo[d] = std::min(lo[d], v[d]); hi[d] = std::max(hi[d], v[d]); }
    };
    double max_semisharp = 0.0;
    for (const auto &kv : cage.edge_crease)
        if (kv.second > 0.0 && kv.second < SHARPNESS_INFINITE) max_semisharp = std::max(max_semisharp, kv.second);
    for (double s : cage.corner_sharp)
        if (s > 0.0 && s < SHARPNESS_INFINITE) max_semisharp = std::max(max_semisharp, s);
    const int stop_floor = std::max(kDeviceLevelMin, int(std::floor(max_semisharp)) + 1);

    Cage rc = cage;
    box(rc, lo, hi);
    for (int l = 0; l < max_level; ++ l) {
        rc = subdivide(rc, scheme, 1);
        Pt nlo, nhi;
        box(rc, nlo, nhi);
        double delta = 0.0, extent = 0.0;
        for (int d = 0; d < 3; ++ d) {
            delta  = std::max(delta,  std::max(std::abs(nlo[d] - lo[d]), std::abs(nhi[d] - hi[d])));
            extent = std::max(extent, nhi[d] - nlo[d]);
        }
        lo = nlo; hi = nhi;
        if (l + 1 >= stop_floor && delta <= extent * 1e-4)
            break;
    }
}

// Vertex -> incident control faces, for 1-ring neighbourhood queries. Build once per cage.
inline std::vector<std::vector<int>> vertex_faces(const Cage &c) {
    std::vector<std::vector<int>> vf(c.nverts());
    for (int f = 0; f < c.nfaces(); ++f)
        for (int k = c.foff[f]; k < c.foff[f + 1]; ++k) vf[c.fvi[k]].push_back(f);
    return vf;
}

// Assemble the sub-cage for a set of core control faces: the core faces first (so their refined
// children lead), then their 1-ring halo (which exactly supports a Catmull-Clark limit patch).
// Carries the face-varying UV and the creases/corners that fall inside. The first core.size() faces
// are the core; the rest is halo. Factored out so a band loop can carry per-face data through it.
inline Cage build_region_subcage(const Cage &c, const std::vector<std::vector<int>> &vf,
                                 const std::vector<int> &core) {
    std::vector<int> faces = core;
    std::map<int, char> seen;
    for (int f : core) seen[f] = 1;
    for (int f : core)
        for (int k = c.foff[f]; k < c.foff[f + 1]; ++k)
            for (int nf : vf[c.fvi[k]])
                if (seen.emplace(nf, 1).second) faces.push_back(nf);

    Cage sub;
    sub.boundary = c.boundary;
    sub.triangle_smooth = c.triangle_smooth;
    std::map<int, int> remap;
    auto vid = [&](int v) {
        auto it = remap.find(v);
        if (it != remap.end()) return it->second;
        int id = int(sub.verts.size());
        remap[v] = id;
        sub.verts.push_back(c.verts[v]);
        return id;
    };
    const bool uv = c.has_uv();
    sub.foff.push_back(0);
    for (int g : faces) {
        for (int k = c.foff[g]; k < c.foff[g + 1]; ++k) {
            sub.fvi.push_back(vid(c.fvi[k]));
            if (uv) sub.fvar.push_back(c.fvar[k]);   // face-varying UV rides with the sub-cage's corners
        }
        sub.foff.push_back(int(sub.fvi.size()));
    }
    // Carry the creases and corners that fall inside the sub-cage.
    const long long nv = c.nverts(), snv = sub.nverts();
    for (const auto &kv : c.edge_crease) {
        auto ia = remap.find(int(kv.first / nv)), ib = remap.find(int(kv.first % nv));
        if (ia != remap.end() && ib != remap.end())
            sub.edge_crease[ekey(ia->second, ib->second, snv)] = kv.second;
    }
    sub.corner_sharp.assign(snv, 0.0);
    if (!c.corner_sharp.empty())
        for (const auto &kv : remap)
            if (kv.first < int(c.corner_sharp.size())) sub.corner_sharp[kv.second] = c.corner_sharp[kv.first];
    return sub;
}

// Catmull-Clark-refine the `core` control faces to `levels`, together with their 1-ring halo. A CC
// patch is supported exactly by its 1-ring, so each core face refines bit-identically to a whole-
// cage refinement; the halo refines wrongly at the cut and is dropped. Returns the core children as
// one shared-indexed cage (no weld), so a Z-band never materializes the whole surface.
inline Cage refine_region(const Cage &c, const std::vector<std::vector<int>> &vf,
                          const std::vector<int> &core, int levels) {
    Cage sub = build_region_subcage(c, vf, core);
    const Cage r = subdivide(sub, "catmullClark", levels);
    // The core faces are sub-cage faces [0, core.size()); their children are the leading output faces
    // -- foff[core.size()] of them at level 1 (one quad per corner), then x4 per further level.
    long long hi = (levels < 1) ? (long long) core.size() : sub.foff[int(core.size())];
    for (int l = 1; l < levels; ++l) hi *= 4;

    // Compact just those child faces and the vertices they use.
    Cage out;
    out.boundary = c.boundary;
    out.triangle_smooth = c.triangle_smooth;
    std::map<int, int> rm;
    auto vid2 = [&](int v) {
        auto it = rm.find(v);
        if (it != rm.end()) return it->second;
        int id = int(out.verts.size());
        rm[v] = id;
        out.verts.push_back(r.verts[v]);
        return id;
    };
    const bool ruv = r.has_uv();
    out.foff.push_back(0);
    for (long long g = 0; g < hi; ++g) {
        for (int k = r.foff[g]; k < r.foff[g + 1]; ++k) {
            out.fvi.push_back(vid2(r.fvi[k]));
            if (ruv) out.fvar.push_back(r.fvar[k]);   // carry the refined face-varying UV to the output
        }
        out.foff.push_back(int(out.fvi.size()));
    }
    return out;
}

}}  // namespace printman::subdiv
