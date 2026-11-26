"""
Referee checks R02 in the actual KLRW algebra of klrw/ (two red strands W_1, W_n; all scalings on).
Written independently of the 02_*.py scripts; the formula is taken from R02_formula.py (pure Python,
from the text of GLUING.md section 2), the diagrams are built from OWN reduced words.

    sage -python proofs/checks/R02_klrw.py product NMAX SAMPLES   # Lemmas 2.2, 2.3 with own reduced words
    sage -python proofs/checks/R02_klrw.py d2poly  NMAX           # Cor. 6.5 model, NON-inverted parameters:
                                                                  #   d^2 = 0 by multiplication in klrw, homogeneity
    sage -python proofs/checks/R02_klrw.py hom     NMAX           # Thm 7.3 by honest linear algebra over QQ:
        all homogeneous x^alpha mu beta (|alpha| <= 3, mu any polynomial monomial in t, u, r of the right klrw
        degree) between S_k and S_l (l < k), chain-map equations computed by multiplication in klrw;
        reports dim(chain maps), rank(boundaries), the rank of Hom_K, and whether the formula's map spans it.
"""
import itertools
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "glue"))
import patches  # noqa: E402
patches.install()
from alg import Alg  # noqa: E402
import importlib  # noqa: E402
F = importlib.import_module("R02_formula")
from sage.matrix.constructor import matrix  # noqa: E402
from sage.modules.free_module_element import vector  # noqa: E402
from sage.rings.rational_field import QQ  # noqa: E402
from sage.rings.integer_ring import ZZ  # noqa: E402


def tok(n, lab):
    if lab == 0:
        return "W1"
    if lab == n + 1:
        return "W%d" % n
    return str(lab)


def toks(n, s):
    return " ".join(tok(n, x) for x in s)


def reduced_word(s, s2, mode="left", rnd=None):
    """positions (1-based) of a reduced word from state s to state s2 (sequences of distinct labels)"""
    cur = list(s)
    pos2 = {x: i for i, x in enumerate(s2)}
    word = []
    while cur != list(s2):
        cand = [i for i in range(len(cur) - 1) if pos2[cur[i]] > pos2[cur[i + 1]]]
        i = cand[0] if mode == "left" else cand[-1] if mode == "right" else rnd.choice(cand)
        cur[i], cur[i + 1] = cur[i + 1], cur[i]
        word.append(i + 1)
    return tuple(word)


class Setup:
    def __init__(self, n, invertible=False):
        self.n = n
        base = [0] + list(range(1, n + 1)) + [n + 1]
        self.A = Alg(["A", n], toks(n, base), invertible=invertible)
        self.D = self.A.D
        self.G = self.A.K.grading_group

    def beta(self, s, s2, mode="left", rnd=None):
        return self.A.word(toks(self.n, s), reduced_word(s, s2, mode, rnd))

    def e(self, s):
        return self.A.e(toks(self.n, s))

    def par(self, nm):
        n = self.n
        if nm == "u1":
            return self.A.u(1)
        if nm == "un":
            return self.A.u(n)
        a, b = map(int, nm[1:].split(","))
        return self.A.t(a, b)

    def xdot(self, c):
        return self.D("x_%d_1" % c)

    def Q(self, x):
        n = self.n
        if x == 0:
            return self.A.u(1) * self.xdot(1)
        if x == n:
            return self.A.u(n) * self.xdot(n)
        return self.A.t(x, x + 1) * self.xdot(x) + self.A.t(x + 1, x) * self.xdot(x + 1)

    def mono(self, e):
        out = self.D.one()
        for nm, ex in e.items():
            out *= self.par(nm) ** ex
        return out


# ------------------------------------------------------------------------------------------ product rule
def check_product(n, samples, seed=3):
    S = Setup(n)
    rnd = random.Random(seed + n)
    labs = list(range(n + 2))
    states = [p for p in itertools.permutations(labs) if p.index(0) < p.index(n + 1)]
    bad = bad_word = 0
    triples = list(itertools.product(states, repeat=3)) if len(states) ** 3 <= samples else \
        [tuple(rnd.choice(states) for _ in range(3)) for _ in range(samples)]
    for s, s1, s2 in triples:
        o = lambda st: tuple(1 if st.index(x) < st.index(x + 1) else 0 for x in range(n + 1))
        D = [x for x in range(n + 1) if o(s1)[x] != o(s)[x] == o(s2)[x]]
        lhs = S.beta(s, s1, "random", rnd) * S.beta(s1, s2, "random", rnd)
        q = S.D.one()
        for x in D:
            q *= S.Q(x)
        rhs = q * S.beta(s, s2, "right")
        bad += (lhs != rhs)
        bad_word += (S.beta(s, s2, "left") != S.beta(s, s2, "random", rnd))
    return len(triples), bad, bad_word


# ------------------------------------------------------------------------------------------ polynomial model
def pi_exp(n, y):
    e = {}
    for x in range(y, n + 1):
        e = F.addexp(e, {F.pname(n, "R%d" % x): 1})
    for x in range(1, y):
        e = F.addexp(e, {F.pname(n, "L%d" % x): 1})
    return e


def poly_coefficient(n, v, w, tag):
    f = [{}]
    for k in range(1, n + 1):
        f.append(F.addexp(f[-1], pi_exp(n, k)))
    e = F.addexp(F.addexp(F.coefficient(n, v, tag), f[v[0]]), f[w[0]], -1)
    assert all(x >= 0 for x in e.values()), (v, tag, e)
    return e


def build_poly(S):
    """formula terms in the polynomial model as klrw elements: dict v -> list of (w, element)"""
    n = S.n
    TT = F.all_terms(n)
    out = {}
    for v, lst in TT.items():
        sv = F.seq(n, v)
        row = []
        for w, tag, sg, e in lst:
            el = (sg * S.mono(poly_coefficient(n, v, w, tag))) * S.beta(sv, F.seq(n, w))
            row.append((w, el))
        out[v] = row
    return out


def vertex_degrees(S, terms):
    """propagate degrees along all terms (P_v -> P_w is an element of degree Delta(v) - Delta(w)); returns
    (Delta, number of inconsistent terms)"""
    n = S.n
    start = (0, tuple("R" * n))
    Dl = {start: S.G.zero()}
    adj = {}
    for v, row in terms.items():
        for w, el in row:
            dg = el.degree(check_if_homogeneous=True)
            adj.setdefault(v, []).append((w, dg, -1))
            adj.setdefault(w, []).append((v, dg, +1))
    stack = [start]
    while stack:
        v = stack.pop()
        for w, dg, s in adj.get(v, []):
            if w not in Dl:
                Dl[w] = Dl[v] + s * dg
                stack.append(w)
    bad = 0
    for v, row in terms.items():
        for w, el in row:
            bad += (Dl[v] - Dl[w] != el.degree(check_if_homogeneous=True))
    return Dl, bad


def check_d2poly(n):
    t0 = time.time()
    S = Setup(n, invertible=False)
    terms = build_poly(S)
    Dl, inhom = vertex_degrees(S, terms)
    # compare with Def. 6.2 shifted by deg f(k), written with klrw's own crossing degrees
    from klrw.framed_dynkin import NodeInFramedQuiver
    node = lambda lab: NodeInFramedQuiver(1, framing=True) if lab == 0 else \
        NodeInFramedQuiver(n, framing=True) if lab == n + 1 else NodeInFramedQuiver(lab)
    E = [S.G.crossing_grading(node(x), node(x + 1)) for x in range(n + 1)]
    f = [{}]
    for k in range(1, n + 1):
        f.append(F.addexp(f[-1], pi_exp(n, k)))
    def deg_mono(e):
        out = S.G.zero()
        for nm, ex in e.items():
            out += ex * S.D.element_degree(S.par(nm), S.G)
        return out
    offs = []
    for v in terms:
        k, s = v
        o = F.obits(n, v)
        Dt = S.G.zero()
        for x in range(n + 1):
            m = o[x] + 2 * (x <= k - 1) + 2 * (1 <= x <= n and s[x - 1] == "C")
            Dt += m * E[x]
        off = Dl[v] - (Dt + deg_mono(f[k]))
        if not any(off == o2 for o2 in offs):
            offs.append(off)
    # d^2
    nz = 0; comps = 0
    for v, row in terms.items():
        acc = {}
        for u, el in row:
            for w, el2 in terms[u]:
                acc[w] = acc.get(w, 0) + el * el2
        for w, val in acc.items():
            comps += 1
            nz += (val != 0)
    return dict(n=n, terms=sum(len(r) for r in terms.values()), inhomogeneous=inhom,
                delta_offsets=len(offs), d2_components=comps, d2_nonzero=nz, seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------ honest Hom
def elem_coords(el, src, tgt):
    out = {}
    for braid, coeff in el.monomial_coefficients().items():
        for exp, c in coeff.dict().items():
            key = (src, tgt, tuple(braid.word()), tuple(exp))
            out[key] = out.get(key, 0) + c
    return {k: v for k, v in out.items() if v}


def check_hom(n, amax=3):
    t0 = time.time()
    S = Setup(n, invertible=False)
    terms = build_poly(S)
    Dl, inhom = vertex_degrees(S, terms)
    assert inhom == 0
    names = [nm for nm in S.D.variable_names() if not nm.startswith("x_")]
    gens = [S.D(nm) for nm in names]
    M = matrix(QQ, [list(S.D.element_degree(g, S.G)._vector_()) for g in gens]).transpose()
    assert M.rank() == len(gens), "parameter degrees not independent"
    dotdeg = [S.D.element_degree(S.xdot(c), S.G) for c in range(1, n + 1)]
    byk = {}
    for v in terms:
        byk.setdefault(v[0], []).append(v)
    hd = F.hdeg
    out = []
    for k in range(1, n + 1):
        cands = {0: [], 1: []}     # hdeg drop -> list of (v, w, element)
        for v in byk[k]:
            for l in range(k):
                for w in byk[l]:
                    dh = hd(v) - hd(w)
                    if dh not in (0, 1):
                        continue
                    b = S.beta(F.seq(n, v), F.seq(n, w))
                    need = Dl[v] - Dl[w] - b.degree(check_if_homogeneous=True)
                    for m in range(amax + 1):
                        for alpha in F.compositions(m, n):
                            rest = need - sum((a * dg for a, dg in zip(alpha, dotdeg)), S.G.zero())
                            try:
                                sol = M.solve_right(vector(QQ, list(rest._vector_())))
                            except ValueError:
                                continue
                            if any(x not in ZZ or x < 0 for x in sol):
                                continue
                            coef = S.D.one()
                            for g, ex in zip(gens, sol):
                                coef *= g ** int(ex)
                            for c, a in enumerate(alpha, 1):
                                coef *= S.xdot(c) ** a
                            cands[dh].append((v, w, coef * b, alpha))
        # chain-map equations: (delta_C g + g d_S) = 0 ; boundaries: delta_C h + h d_S (h hdeg-preserving)
        pred = {}
        for v in byk[k]:
            for u, el in terms[v]:
                if u[0] == k:
                    pred.setdefault(u, []).append((v, el))
        def image(v, w, el):
            vec = {}
            for x, el2 in terms[w]:                      # g, then the differential of C_{k-1}
                for kk, c in elem_coords(el * el2, v, x).items():
                    vec[kk] = vec.get(kk, 0) + c
            for y, el0 in pred.get(v, []):               # the differential of S_k, then g
                for kk, c in elem_coords(el0 * el, y, w).items():
                    vec[kk] = vec.get(kk, 0) + c
            return {kk: c for kk, c in vec.items() if c}
        imgs = [image(c[0], c[1], c[2]) for c in cands[1]]
        keys = sorted(set(kk for im in imgs for kk in im), key=str)
        kidx = {kk: i for i, kk in enumerate(keys)}
        Mz = matrix(QQ, len(keys), len(imgs), {(kidx[kk], j): c for j, im in enumerate(imgs) for kk, c in im.items()},
                    sparse=True)
        Z = Mz.right_kernel()
        # boundaries: express d h + h d in the coordinates of the candidates (it is a degree-0 map S_k -> C_{k-1})
        cidx = {}
        for j, (v, w, el, al) in enumerate(cands[1]):
            for kk, c in elem_coords(el, v, w).items():
                cidx.setdefault(kk, []).append((j, c))
        bnd = []
        for (v, w, el, al) in cands[0]:
            im = image(v, w, el)
            bnd.append(im)
        rank_b = 0
        if bnd:
            # boundaries live in the same coordinate space as the chain maps (components v -> x)
            allkeys = sorted(set(kk for b_ in bnd for kk in b_) | set(cidx), key=str)
            aidx = {kk: i for i, kk in enumerate(allkeys)}
            Mb = matrix(QQ, len(allkeys), len(bnd), {(aidx[kk], j): c for j, b_ in enumerate(bnd) for kk, c in b_.items()},
                        sparse=True)
            rank_b = Mb.rank()
        # the formula's gluing map as a vector in the candidate basis
        fvec = [0] * len(cands[1])
        formula_keys = {}
        for v in byk[k]:
            for w, el in terms[v]:
                if w[0] < k:
                    for kk, c in elem_coords(el, v, w).items():
                        formula_keys[kk] = c
        cand_keys = {}
        for j, (v, w, el, al) in enumerate(cands[1]):
            co = elem_coords(el, v, w)
            assert len(co) == 1
            (kk, c), = co.items()
            cand_keys[kk] = (j, c)
        missing = [kk for kk in formula_keys if kk not in cand_keys]
        for kk, c in formula_keys.items():
            if kk in cand_keys:
                j, c0 = cand_keys[kk]
                fvec[j] = QQ(c) / QQ(c0)
        in_kernel = (Mz * vector(QQ, fvec)).is_zero()
        ndots = sum(1 for c in cands[1] if sum(c[3]))
        out.append(dict(k=k, chain_candidates=len(cands[1]), with_dots=ndots, homotopy_candidates=len(cands[0]),
                        dim_cycles=Z.dimension(), rank_boundaries=rank_b, rank_Hom=Z.dimension() - rank_b,
                        formula_is_cycle=in_kernel, formula_terms_missing_from_ansatz=len(missing)))
    return out, round(time.time() - t0, 1)


if __name__ == "__main__":
    what = sys.argv[1]
    N = int(sys.argv[2])
    if what == "product":
        smp = int(sys.argv[3]) if len(sys.argv) > 3 else 500
        for n in range(1, N + 1):
            print("n=%d: triples %d, product-rule mismatches %d, word-dependence mismatches %d" % ((n,) + check_product(n, smp)),
                  flush=True)
    elif what == "d2poly":
        for n in range(1, N + 1):
            print(check_d2poly(n), flush=True)
    elif what == "hom":
        lo = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        for n in range(lo, N + 1):
            res, sec = check_hom(n)
            print("n=%d (%ss):" % (n, sec), flush=True)
            for r in res:
                print("   ", r, flush=True)
