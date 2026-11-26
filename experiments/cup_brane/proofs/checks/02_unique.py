"""
Uniqueness of the gluing maps (proofs/02_omega1_ibrane.md, Theorem 7.5), checked by linear
algebra in the R-model, pure Python.

    python3 proofs/checks/02_unique.py N_MAX

For every k = 1..n: unknown integers c_t, one for every gluing term t of the formula out of the
envelope S_k (all target envelopes l < k).  By Corollary 7.3 every homogeneous map S_k -> C_{k-1}
of the degree of the differential is  sum_t c_t * t  (and there are no homotopies).  It is a chain
map  S_k -> C_{k-1}  (C_{k-1} = the formula on S_0 .. S_{k-1}) iff the components out of S_k of
(delta_{<k} + d_k + sum_t c_t t)^2 vanish.  We solve this linear system over Q in the R-model with
q_1..q_n free and q_0 := -sum_{x>=1} (-1)^x q_x (so W = 0; this ring embeds into the actual
coefficient ring, q_x = T_x Q_x, see section 5) and report the dimension of the solution space
(expected: 1, spanned by c = 1), and separately for the maps S_k -> S_{k-1} alone (U1) and for the
maps S_k -> S_{k-d}, d >= 2, that are chain maps on their own (U2, expected 0).
"""
import itertools
import os
import sys
from collections import defaultdict
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
R = importlib.import_module("02_rmodel")


def subst_W(n, p):
    """Replace q_0 by sum_{x>=1} (-1)^(x+1) q_x in a q-polynomial; return dict over exponents of q_1..q_n."""
    out = defaultdict(int)
    for e, c in p.items():
        e0 = e[0]
        rest = list(e[1:])
        # expand (sum_x s_x q_x)^e0
        terms = {tuple(rest): c}
        for _ in range(e0):
            new = defaultdict(int)
            for ee, cc in terms.items():
                for x in range(1, n + 1):
                    e2 = list(ee); e2[x - 1] += 1
                    new[tuple(e2)] += cc * (-1) ** (x + 1)
            terms = new
        for ee, cc in terms.items():
            out[ee] += cc
    return {k: v for k, v in out.items() if v}


def nullity(rows, nvars):
    """rows: list of dict var -> Fraction. Returns dimension of the solution space."""
    piv = {}
    rank = 0
    mat = [dict(r) for r in rows if r]
    basis = []
    for r in mat:
        r = {k: Fraction(v) for k, v in r.items() if v}
        for p, br in basis:
            if p in r:
                f = r[p]
                for k2, v2 in br.items():
                    r[k2] = r.get(k2, 0) - f * v2
                r = {k2: v2 for k2, v2 in r.items() if v2}
        if r:
            p = min(r)
            f = r[p]
            r = {k2: v2 / f for k2, v2 in r.items()}
            # reduce existing basis
            nb = []
            for p2, br in basis:
                if p in br:
                    g = br[p]
                    br = {k2: br.get(k2, 0) - g * r.get(k2, 0) for k2 in set(br) | set(r)}
                    br = {k2: v2 for k2, v2 in br.items() if v2}
                nb.append((p2, br))
            basis = nb + [(p, r)]
    return nvars - len(basis), basis


def run(n):
    verts = list(R.rmodel_vertices(n))
    terms = {v: R.rmodel_terms(n, *v) for v in verts}
    results = []
    for k in range(1, n + 1):
        unknown = []          # (v, index into terms[v])
        for v in verts:
            if v[0] != k:
                continue
            for i, (l, R2, cf, tag) in enumerate(terms[v]):
                if l < k:
                    unknown.append((v, i))
        idx = {u: j for j, u in enumerate(unknown)}

        def equations(allowed):
            """allowed(l): which unknown target envelopes take part; others fixed to 0."""
            comp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # (v,w) -> mono -> var -> coeff
            for v in verts:
                if v[0] != k:
                    continue
                for i, (l, R1, c1, tag1) in enumerate(terms[v]):
                    u = (l, R1)
                    if l == k:   # internal first, then an unknown
                        for j2, (l2, R2, c2, tag2) in enumerate(terms[u]):
                            if l2 < k and allowed(l2):
                                p = R.qmul(c1, c2)
                                for mo, cc in subst_W(n, p).items():
                                    comp[(v, (l2, R2))][mo][idx[(u, j2)]] += cc
                    elif allowed(l):  # unknown first, then the fixed differential of C_{k-1}
                        for (l2, R2, c2, tag2) in terms[u]:
                            p = R.qmul(c1, c2)
                            for mo, cc in subst_W(n, p).items():
                                comp[(v, (l2, R2))][mo][idx[(v, i)]] += cc
            rows = []
            for key, monos in comp.items():
                for mo, row in monos.items():
                    rows.append({a: b for a, b in row.items() if b})
            return rows

        rows = equations(lambda l: True)
        dim, basis = nullity(rows, len(unknown))
        # is c = 1 a solution?
        ones_ok = all(sum(r.values()) == 0 for r in rows)
        # U1 (d = 1) and U2 (d >= 2): maps supported on S_k -> S_{k-d} that are chain maps on their own
        dims2 = []
        for d in range(1, k + 1):
            sub = [j for j, (v, i) in enumerate(unknown) if terms[v][i][0] == k - d]
            comp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for v in verts:
                if v[0] != k:
                    continue
                for i, (l, R1, c1, tag1) in enumerate(terms[v]):
                    u = (l, R1)
                    if l == k:
                        for j2, (l2, R2, c2, tag2) in enumerate(terms[u]):
                            if l2 == k - d:
                                for mo, cc in subst_W(n, R.qmul(c1, c2)).items():
                                    comp[(v, (l2, R2))][mo][idx[(u, j2)]] += cc
                    elif l == k - d:
                        for (l2, R2, c2, tag2) in terms[u]:
                            if l2 == l:     # internal differential of S_{k-d}
                                for mo, cc in subst_W(n, R.qmul(c1, c2)).items():
                                    comp[(v, (l2, R2))][mo][idx[(v, i)]] += cc
            rws = [{a: b for a, b in row.items() if b} for monos in comp.values() for row in monos.values()]
            dd, _ = nullity([r for r in rws if r], len(sub))
            dims2.append(dd)
        results.append((k, len(unknown), dim, ones_ok, dims2[0], dims2[1:]))
    return results


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for n in range(1, N + 1):
        for k, nu, dim, ones_ok, dim1, dims2 in run(n):
            print("n=%d k=%d: %d unknown gluing coefficients; solution space of the chain-map equations has dim %d"
                  " (c = 1 is a solution: %s); U1: chain maps S_k -> S_{k-1} on formula support: dim %d;"
                  " U2: chain maps S_k -> S_{k-d}, d>=2: dims %s" % (n, k, nu, dim, ones_ok, dim1, dims2), flush=True)
