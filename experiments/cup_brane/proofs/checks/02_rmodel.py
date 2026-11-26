"""
The "R-model" of the omega_1 I-brane (proofs/02_omega1_ibrane.md, sections 3-5), pure Python.

    python3 proofs/checks/02_rmodel.py N_MAX

Checks, for n = 1..N_MAX:
  (a) [Prop. 3.4 + Lemma 4.2] the scalar complex of the I-brane (GLUING.md formula, bigon rule of
      Lemma 2.3, reference bits b = 1) becomes, after the diagonal change of basis
      f_v = sigma(v) phi(v) e_v, exactly the R-model with q_x = T_x Q_x  (entry by entry, as
      polynomials in the dots with Laurent coefficients);
  (b) [Theorem 5.1] in the R-model over Z[q_0..q_n] (free variables):  delta^2 = W * H with
      W = sum_x (-1)^x q_x  and  H = sum_{j<k} (-1)^(j+1) e_j (x) N_{j,k};
  (c) [Lemma 5.3, 5.4] the block identities (Psi = 0, Inner = -W_[0,D] N_{0,D}) for D <= N_MAX+?;
  (d) the closed formula for sigma(v) (vertex signs) and phi(v).

R-model: vertex (k, R), 0 <= k <= n, R subset {1..n}  (R = A u C).  Lambda(r_1..r_n), order
r_1 < ... < r_n.  delta(e_k (x) m) = e_k (x) d_k m + sum_{l<k} e_l (x) Z_{l,k} m, with
   d_k     = sum_{c<=k} eps_c + sum_{c>k} q_c iota_c
   Z_{l,k} = -q_l iota_k Pi[l+1,k] + ((-1)^(d+1) eps_{l+1} + E_{l,k} iota_k eps_{l+1}) Pi[l+2,k]
   E_{l,k} = sum_{c<=l} (-1)^(l-c) q_{c-1} iota_c  -  sum_{c>k} (-1)^(c-l) eps_c      (d = k-l).
"""
import itertools
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
M = importlib.import_module("02_model")


# ----------------------------------------------------------------------------- q-polynomials
# a q-polynomial is a dict: exponent tuple (length n+1) -> int
def qmono(n, x):
    e = [0] * (n + 1); e[x] = 1
    return {tuple(e): 1}


def qconst(n, c=1):
    return {(0,) * (n + 1): c} if c else {}


def qmul(p, q):
    out = defaultdict(int)
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            out[tuple(a + b for a, b in zip(e1, e2))] += c1 * c2
    return {k: v for k, v in out.items() if v}


def qadd(p, q, s=1):
    out = defaultdict(int, p)
    for k, v in q.items():
        out[k] += s * v
    return {k: v for k, v in out.items() if v}


def qscale(p, s):
    return {k: s * v for k, v in p.items() if s * v}


# ----------------------------------------------------------------------------- Clifford on basis sets
def eps(c, R):
    if c in R:
        return None
    return R | {c}, (-1) ** sum(1 for x in R if x < c)


def iota(c, R):
    if c not in R:
        return None
    return R - {c}, (-1) ** sum(1 for x in R if x < c)


def run(ops, R):
    """ops: list of ('e'|'i'|'P1'|'P0', c), applied in list order (first = rightmost factor)."""
    s = 1
    for op, c in ops:
        if op == "e":
            r = eps(c, R)
        elif op == "i":
            r = iota(c, R)
        elif op == "P1":
            r = (R, 1) if c in R else None
        else:
            r = (R, 1) if c not in R else None
        if r is None:
            return None
        R, t = r
        s *= t
    return R, s


def Pi(a, b):
    return [("P1", c) for c in range(a, b + 1)]


# ----------------------------------------------------------------------------- the R-model
def rmodel_terms(n, k, R, inner=None):
    """
    All terms out of (k, R): list of (l, R', q-polynomial coefficient incl. sign, tag).
    ``inner`` = (j, kk): only colours in (j, kk] exist; outer E-parts dropped (block identities).
    """
    out = []
    lo, hi = (1, n) if inner is None else (inner[0] + 1, inner[1])
    cols = range(lo, hi + 1)
    # internal
    for c in cols:
        if c <= k:
            r = run([("e", c)], R)
            if r:
                out.append((k, r[0], qconst(n, r[1]), ("int", c)))
        else:
            r = run([("i", c)], R)
            if r:
                out.append((k, r[0], qscale(qmono(n, c), r[1]), ("int", c)))
    lmin = 0 if inner is None else inner[0]
    for l in range(lmin, k):
        d = k - l
        # (i)
        r = run(Pi(l + 1, k) + [("i", k)], R)
        if r:
            out.append((l, r[0], qscale(qmono(n, l), -r[1]), ("i", d)))
        # (ii)
        r = run(Pi(l + 2, k) + [("e", l + 1)], R)
        if r:
            out.append((l, r[0], qconst(n, (-1) ** (d + 1) * r[1]), ("ii", d)))
        # (iii)
        base = Pi(l + 2, k) + [("e", l + 1), ("i", k)]
        for c in cols:
            if l < c <= k:
                continue
            if inner is not None and l == inner[0] and k == inner[1]:
                continue                      # E_{j,k} is purely outer
            if c <= l:
                r = run(base + [("i", c)], R)
                coef = qscale(qmono(n, c - 1), (-1) ** (l - c))
            else:
                r = run(base + [("e", c)], R)
                coef = qconst(n, -(-1) ** (c - l))
            if r:
                out.append((l, r[0], qscale(coef, r[1]), ("iii", d, c)))
    return out


def rmodel_vertices(n):
    for k in range(n + 1):
        for s in range(n + 1):
            for R in itertools.combinations(range(1, n + 1), s):
                yield (k, frozenset(R))


def square(n, verts, termfn):
    """delta^2 as dict (src, tgt) -> q-polynomial."""
    out = defaultdict(dict)
    cache = {}
    def T(v):
        if v not in cache:
            cache[v] = termfn(*v)
        return cache[v]
    for v in verts:
        for (l, R1, c1, tag1) in T(v):
            for (j, R2, c2, tag2) in T((l, R1)):
                key = (v, (j, R2))
                out[key] = qadd(out[key], qmul(c1, c2))
    return {key: p for key, p in out.items() if p}


def W(n, lo=0, hi=None):
    hi = n if hi is None else hi
    p = {}
    for x in range(lo, hi + 1):
        p = qadd(p, qscale(qmono(n, x), (-1) ** x))
    return p


def N_op(n, j, k, R):
    """N_{j,k} = iota_k eps_{j+1} Pi[j+2,k] on r_R."""
    return run(Pi(j + 2, k) + [("e", j + 1), ("i", k)], R)


def check_universal(n):
    """(b): delta^2 = W * H in the R-model."""
    sq = square(n, list(rmodel_vertices(n)), lambda k, R: rmodel_terms(n, k, R))
    expect = {}
    Wn = W(n)
    for (k, R) in rmodel_vertices(n):
        for j in range(k):
            r = N_op(n, j, k, R)
            if r:
                expect[((k, R), (j, r[0]))] = qscale(Wn, (-1) ** (j + 1) * r[1])
    ok = set(sq) == set(expect) and all(sq[key] == expect[key] for key in sq)
    return ok, len(sq)


def check_blocks(D):
    """(c): Lemma 5.3 (Psi = 0) and Lemma 5.4 (Inner) on the block colours 1..D, j = 0, k = D."""
    n = D
    # Inner: d_0 Z_{0,D} + Z_{0,D} d_D + sum_l Z_{0,l} Z_{l,D}, restricted to colours 1..D, no outer parts
    verts = [(k, frozenset(R)) for k in range(D + 1) for s in range(D + 1)
             for R in itertools.combinations(range(1, D + 1), s)]
    def tf(k, R):
        return [t for t in rmodel_terms(n, k, R, inner=(0, D))]
    sq = square(n, [v for v in verts if v[0] == D], tf)
    inner = {key: p for key, p in sq.items() if key[1][0] == 0}
    expect = {}
    for R in [frozenset(S) for s in range(D + 1) for S in itertools.combinations(range(1, D + 1), s)]:
        r = N_op(n, 0, D, R)
        if r:
            expect[((D, R), (0, r[0]))] = qscale(W(n, 0, D), -r[1])
    ok_inner = set(inner) == set(expect) and all(inner[k_] == expect[k_] for k_ in inner)
    # Psi = -d_0 N_{0,D} + N_{0,D} d_D + sum_l [N_{0,l} Z_{l,D} - (-1)^l Z_{0,l} N_{l,D}]
    psi = defaultdict(dict)
    for R in [frozenset(S) for s in range(D + 1) for S in itertools.combinations(range(1, D + 1), s)]:
        src = (D, R)
        # N_{0,D} d_D  (d_D first)
        for (k2, R1, c1, tag) in tf(D, R):
            if tag[0] == "int":
                r = N_op(n, 0, D, R1)
                if r:
                    psi[(src, r[0])] = qadd(psi[(src, r[0])], qscale(c1, r[1]))
        # - d_0 N_{0,D}
        r = N_op(n, 0, D, R)
        if r:
            for (k2, R2, c2, tag) in tf(0, r[0]):
                if tag[0] == "int":
                    psi[(src, R2)] = qadd(psi[(src, R2)], qscale(c2, -r[1]))
        for l in range(1, D):
            # N_{0,l} Z_{l,D}
            for (l2, R1, c1, tag) in tf(D, R):
                if l2 == l and tag[0] != "int":
                    r = N_op(n, 0, l, R1)
                    if r:
                        psi[(src, r[0])] = qadd(psi[(src, r[0])], qscale(c1, r[1]))
            # -(-1)^l Z_{0,l} N_{l,D}
            r = N_op(n, l, D, R)
            if r:
                for (j2, R2, c2, tag) in tf(l, r[0]):
                    if j2 == 0 and tag[0] != "int":
                        psi[(src, R2)] = qadd(psi[(src, R2)], qscale(c2, -(-1) ** l * r[1]))
    ok_psi = all(not p for p in psi.values())
    return ok_inner, ok_psi


# ----------------------------------------------------------------------------- comparison with the I-brane
def sigma(n, v):
    """Vertex sign of Lemma 4.2: (-1)^(C(|A|,2) + C(|C|,2) + sum(A) + |B_L| |C|)."""
    k, A, C = v
    a, c = len(A), len(C)
    bl = k - a
    return (-1) ** (a * (a - 1) // 2 + c * (c - 1) // 2 + sum(A) + bl * c)


def T_mono(n, x):
    """T_x = prod_{y=1..x} tau_y as a Laurent monomial (frozenset)."""
    return M.tau_power(n, 1, x, 1) if x >= 1 else M.ONE


def phi(n, v):
    """phi(v) = psi(k) prod_{c in C} T_c,   psi(k) = prod_{c=1}^{k-1} T_c."""
    k, A, C = v
    m = M.ONE
    for c in list(range(1, k)) + sorted(C):
        m = M.mono_mul(m, T_mono(n, c))
    return m


def mono_inv(m):
    return frozenset((a, -b) for a, b in m)


def q_to_poly(n, qp):
    """Substitute q_x = T_x Q_x into a q-polynomial; result is a polynomial in the dots (02_model format)."""
    out = {}
    for e, c in qp.items():
        p = M.poly_const(n, M.ONE, c)
        for x, ex in enumerate(e):
            for _ in range(ex):
                p = M.poly_mul(p, M.poly_mul(M.Q(n, x), M.poly_const(n, T_mono(n, x))))
        out = M.poly_add(out, p)
    return out


def check_iso(n):
    """(a): scalar complex (b = 1) in the basis f_v = sigma(v) phi(v) e_v equals the R-model at q = TQ."""
    bad = 0; count = 0
    rterms = {}
    for (k, R) in rmodel_vertices(n):
        for (l, R2, cf, tag) in rmodel_terms(n, k, R):
            rterms[((k, R), (l, R2))] = cf
    seen = set()
    for t in M.all_terms(n):
        v, w = t.src, t.tgt
        ov, ow = M.order_bits(n, v), M.order_bits(n, w)
        flips = [x for x in range(n + 1) if ov[x] == 0 and ow[x] == 1]
        p = M.poly_const(n, t.coeff, t.sign)
        for x in flips:
            p = M.poly_mul(p, M.Q(n, x))
        # change of basis: f_v = s(v) phi(v) e_v  =>  entry * s(v) s(w) * phi(v)/phi(w)
        p = M.poly_mul(p, M.poly_const(n, M.mono_mul(phi(n, v), mono_inv(phi(n, w))), sigma(n, v) * sigma(n, w)))
        key = ((v[0], v[1] | v[2]), (w[0], w[1] | w[2]))
        seen.add(key)
        count += 1
        if key not in rterms or q_to_poly(n, rterms[key]) != p:
            bad += 1
    missing = len(set(rterms) - seen)
    return count, bad, missing


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for n in range(1, N + 1):
        count, bad, missing = check_iso(n)
        ok, ncomp = check_universal(n)
        print("n=%d: I-brane terms %d, differing from the R-model %d, R-model terms not in the I-brane %d;"
              " delta^2 = W*H in the universal R-model: %s (%d non-zero components)"
              % (n, count, bad, missing, ok, ncomp), flush=True)
    for D in range(1, N + 3):
        print("block identities, D=%d: Inner = -W_[0,D] N_{0,D}: %s;  Psi = 0: %s" % ((D,) + check_blocks(D)), flush=True)
