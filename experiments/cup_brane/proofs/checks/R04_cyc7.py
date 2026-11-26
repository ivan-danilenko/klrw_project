"""
Referee re-derivation of Theorem 9.1 of proofs/04_first_order.md (word 1221122, A_2, red 2).

    sage -python proofs/checks/R04_cyc7.py b      (b): r-degree parts of d_T^2 (klrw basis braids, and own words)
    sage -python proofs/checks/R04_cyc7.py c      (c): dot-free r-degree-2 cycles and the rank 4 -> 5 claim
    sage -python proofs/checks/R04_cyc7.py d      (d): (d_T + X)^2 = 0 with X typed in from the text; kernels at the 4 orders
    sage -python proofs/checks/R04_cyc7.py z1     dimension of the dot-free first-order cycle space (claimed 106)

Own code: homogeneous elements are enumerated with R04_comb.W.solve (own grading); maps on the
cube, commutators and vectorisation are written here.  The tree formula (signs) is formula.Stack;
the basis braid is fast.braid_for (Definition B.10 of 01_foundations.md), as in the paper.
"""
import sys, os, itertools, time
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube  # noqa: E402
from fast import braid_for  # noqa: E402
from formula import Stack  # noqa: E402
from R04_comb import W, ident, transp  # noqa: E402
from sage.all import matrix, QQ, vector  # noqa: E402

CT, RED, WORD = ["A", 2], 2, (1, 2, 2, 1, 1, 2, 2)


class Ctx:
    def __init__(self, own_words=False):
        P = [(RED, WORD)]
        self.K = make_algebra(CT, P)
        self.cube = Cube(self.K, P, build=False)
        self.D = self.K.base()
        self.names = list(self.D.variable_names())
        self.rpos = [i for i, n in enumerate(self.names) if n.startswith("r_")]
        self.ppos = [i for i, n in enumerate(self.names) if not n.startswith("x_")]
        self.Wd = W(CT, RED, WORD)
        self.st = Stack(CT, WORD)
        self.own = own_words
        self._b = {}
        self.V = list(self.Wd.subsets())

    def eps(self, I):
        return tuple(1 if s + 1 in I else 0 for s in range(self.Wd.m))

    def basis(self, I, J, sigma):
        """sigma: dict slot -> slot"""
        key = (I, J, tuple(sorted(sigma.items())))
        if key not in self._b:
            if self.own:
                w, _ = self.Wd.bubble_word(I, J, sigma)
                self._b[key] = self.K.element_from_braid(tuple(w), left_state=self.cube.state(self.eps(I)))
            else:
                _, _, braid = braid_for(self.cube, I, J, tuple(sigma[s] for s in range(1, self.Wd.m + 1)))
                self._b[key] = self.K.monomial(braid)
        return self._b[key]

    def mono(self, n, m):
        e = self.D.one()
        for c, p in n.items():
            e *= self.D("r_%d" % c) ** p
        for (a, b), p in m.items():
            e *= self.D("t_%d_%d" % (a, b)) ** p
        return e

    def monot(self, mono):
        e = self.D.one()
        for nm, p in mono:
            e *= self.D(nm) ** p
        return e

    def rdeg(self, e):
        return sum(e[p] for p in self.rpos)

    def tree(self, maxpairs=99, onlypairs=None):
        d = {}
        for I in self.V:
            row = defaultdict(lambda: self.K.zero())
            for (J, sigma, mono, sign) in self.st.terms_from(I):
                k = sum(p for nm, p in mono if nm.startswith("r_"))
                if k > maxpairs or (onlypairs is not None and k != onlypairs):
                    continue
                sg = {s: sigma[s - 1] for s in range(1, self.Wd.m + 1)}
                row[J] += (self.D(sign) * self.monot(mono)) * self.basis(I, J, sg)
            d[I] = dict(row)
        return d


def compose(C, f, g):
    res = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for I, row in f.items():
        for J, x in row.items():
            for K, y in g.get(J, {}).items():
                res[I][K] += x * y
    return res


def anticomm(C, d, I, J, el):
    """(d . eta + eta . d) for eta = el in the single component I -> J: dict (I', K') -> element"""
    out = defaultdict(lambda: C.K.zero())
    for Ip, row in d.items():
        if J and False:
            pass
        x = row.get(I)
        if x is not None:
            out[Ip, J] += x * el
    for K, y in d.get(J, {}).items():
        out[I, K] += el * y
    return out


def vectorize(C, comp, keep, idx, grow=True):
    vec = {}
    for (I, K), x in comp.items():
        for braid, coeff in x:
            for e, c in coeff.dict().items():
                if not keep(e):
                    continue
                key = (I, K, braid, e)
                if key not in idx:
                    if not grow:
                        raise KeyError
                    idx[key] = len(idx)
                vec[idx[key]] = vec.get(idx[key], 0) + int(c)
    return {k: v for k, v in vec.items() if v}


def candidates(C, hdeg=-1, rdeg=None, dotfree=True, monofilter=None):
    out = []
    Wd = C.Wd
    bij = list(Wd.bijections())
    for I in C.V:
        for J in C.V:
            if len(J) != len(I) + hdeg:
                continue
            for sigma in bij:
                for (n, m, l, a) in Wd.solve(Wd.defect(I, J, sigma)):
                    if l:
                        continue
                    if dotfree and a:
                        continue
                    if rdeg is not None and sum(n.values()) != rdeg:
                        continue
                    if monofilter is not None and not monofilter(n, m):
                        continue
                    out.append((I, J, sigma, n, m, a))
    return out


def dot_monomials(C, I, a):
    """all dot monomials with a[c] dots on the strands of colour c in e_I (klrw names x_c_k)"""
    labs = [l for l in C.Wd.labels(I) if l != "W"]
    per = []
    for c, k in sorted(a.items()):
        nstr = sum(1 for l in labs if C.Wd.c(l) == c)
        vars_ = [C.D("x_%d_%d" % (c, i)) for i in range(1, nstr + 1)]
        per.append([prod_(comb) for comb in itertools.combinations_with_replacement(vars_, k)])
    res = []
    for combo in itertools.product(*per):
        e = C.D.one()
        for f in combo:
            e *= f
        res.append(e)
    return res


def prod_(lst):
    e = 1
    for f in lst:
        e = e * f
    return e


def part_b():
    for own in (False, True):
        C = Ctx(own_words=own)
        t0 = time.time()
        dT = C.tree()
        sq = compose(C, dT, dT)
        byr = Counter()
        comps = set()
        dotted = Counter()
        for I, row in sq.items():
            for K, x in row.items():
                for braid, coeff in x:
                    for e, c in coeff.dict().items():
                        rd = C.rdeg(e)
                        byr[rd] += 1
                        if rd == 3:
                            comps.add((tuple(sorted(I)), tuple(sorted(K))))
                            ndots = sum(e[p] for p in range(len(C.names)) if C.names[p].startswith("x_"))
                            dotted[ndots] += 1
        print("(b) %s: tree terms %d; d_T^2 coefficients by r-degree %s; r^3 part: components %s, by #dots %s [%.0fs]" % (
            "OWN bubble words" if own else "klrw basis braids", len(C.st.all_terms()), dict(sorted(byr.items())),
            sorted(comps), dict(dotted), time.time() - t0))


def part_c():
    C = Ctx()
    t0 = time.time()
    d0 = C.tree(0)
    d01 = C.tree(1)
    dT = C.tree()
    sq = compose(C, dT, dT)
    sq = {(I, K): x for I, row in sq.items() for K, x in row.items()}
    idx3 = {}
    obs = vectorize(C, sq, lambda e: C.rdeg(e) == 3, idx3)
    low = vectorize(C, sq, lambda e: C.rdeg(e) <= 2, {})
    print("(c) (d_T^2): r-degree <= 2 coefficients:", len(low), "; r-degree-3 coefficients:", len(obs))
    cands = candidates(C, -1, 2, True)
    print("    dot-free r-degree-2 homogeneous candidates:", len(cands))
    idx2 = {}
    cols = []
    els = []
    for (I, J, sigma, n, m, a) in cands:
        el = C.mono(n, m) * C.basis(I, J, sigma)
        els.append((I, J, el))
        cols.append(vectorize(C, anticomm(C, d0, I, J, el), lambda e: C.rdeg(e) == 2, idx2))
    M = matrix(QQ, max(len(idx2), 1), len(cols), {(r_, c_): v for c_, col in enumerate(cols) for r_, v in col.items()})
    ker = M.right_kernel().basis()
    print("    kernel of [delta_0, -] mod r^3 on them: dim", len(ker))
    vecs = []
    from sage.all import lcm
    ker = [v * lcm([QQ(c).denominator() for c in v.list()]) for v in ker]
    for v in ker:
        tot = defaultdict(lambda: C.K.zero())
        for i, c in v.dict().items():
            I, J, el = els[i]
            for key, x in anticomm(C, d01, I, J, el).items():
                tot[key] += C.D(int(c)) * x
        vecs.append(vectorize(C, tot, lambda e: C.rdeg(e) == 3, idx3))
        low2 = vectorize(C, tot, lambda e: C.rdeg(e) <= 2, {})
        if low2:
            print("    WARNING: kernel element has r-degree <= 2 part in [delta_0+delta_1, kappa]", len(low2))
    n = len(idx3)
    Mk = matrix(QQ, n, len(vecs), {(r_, c_): v for c_, col in enumerate(vecs) for r_, v in col.items()})
    ob = vector(QQ, n, obs)
    r1 = Mk.rank()
    r2 = Mk.augment(ob.column()).rank()
    print("    rank of r^3-parts of [delta_0+delta_1, kappa]: %d; with (d_T^2)_3: %d  -> %s [%.0fs]" % (
        r1, r2, "NOT removable" if r2 > r1 else "removable", time.time() - t0))
    for v in ker:
        print("    kernel vector:", {(tuple(sorted(els[i][0])), tuple(sorted(els[i][1]))): c for i, c in v.dict().items()})


def X_elements(C):
    I0, J0 = frozenset({4, 5, 6, 7}), frozenset({1, 2, 3})
    D = C.D
    r1, r2, t12, t21 = D("r_1"), D("r_2"), D("t_1_2"), D("t_2_1")
    x = lambda c, k: D("x_%d_%d" % (c, k))
    s_cyc = {1: 4, 2: 7, 3: 6, 4: 1, 5: 5, 6: 3, 7: 2}
    s1 = {1: 4, 2: 6, 3: 7, 4: 1, 5: 5, 6: 3, 7: 2}
    s2 = {1: 4, 2: 7, 3: 6, 4: 1, 5: 5, 6: 2, 7: 3}
    s3 = {1: 5, 2: 7, 3: 6, 4: 1, 5: 4, 6: 3, 7: 2}
    Xdot = -(r1 * r2 ** 2 * t12 * t21 ** 2) * (t21 * (x(2, 1) + x(2, 2) + x(2, 3) + x(2, 4)) +
                                             t12 * (x(1, 1) + x(1, 2) + 2 * x(1, 3))) * C.basis(I0, J0, s_cyc)
    Xr = (r1 * r2 ** 3 * t12 * t21 ** 3) * C.basis(I0, J0, s1) - (r1 * r2 ** 3 * t12 * t21 ** 3) * C.basis(I0, J0, s2) \
        - (r1 ** 2 * r2 ** 2 * t12 ** 2 * t21 ** 2) * C.basis(I0, J0, s3)
    return I0, J0, Xdot, Xr


def part_d():
    C = Ctx()
    t0 = time.time()
    dT = C.tree()
    I0, J0, Xdot, Xr = X_elements(C)
    for name, X in (("X_dot + X_r", Xdot + Xr), ("X_dot only", Xdot), ("no X", C.K.zero())):
        d = {I: dict(row) for I, row in dT.items()}
        d[I0][J0] = d[I0].get(J0, C.K.zero()) + X
        sq = compose(C, d, d)
        nz = sum(1 for row in sq.values() for x in row.values() if not x.is_zero())
        print("(d) d_T + (%s): non-zero components of d^2: %d" % (name, nz))
    # is the I0 -> J0 component of d_T empty (no tree term there)?
    print("    tree formula has a term I0 -> J0:", J0 in dT[I0])
    # kernels at the four orders: complete ansatz with dots, linear map b -> parameter-free part of [delta_0, b]
    d0 = C.tree(0)
    orders = {"r1 r2^2 t12 t21^3": ({1: 1, 2: 2}, {(1, 2): 1, (2, 1): 3}),
              "r1 r2^2 t12^2 t21^2": ({1: 1, 2: 2}, {(1, 2): 2, (2, 1): 2}),
              "r1 r2^3 t12 t21^3": ({1: 1, 2: 3}, {(1, 2): 1, (2, 1): 3}),
              "r1^2 r2^2 t12^2 t21^2": ({1: 2, 2: 2}, {(1, 2): 2, (2, 1): 2})}
    for name, (n0, m0) in orders.items():
        cands = candidates(C, -1, None, False, monofilter=lambda n, m: n == n0 and m == m0)
        idx = {}
        cols = []
        for (I, J, sigma, n, m, a) in cands:
            for dm in dot_monomials(C, I, a):
                el = dm * C.basis(I, J, sigma)
                cols.append(vectorize(C, anticomm(C, d0, I, J, el),
                                      lambda e: all(e[p] == 0 for p in C.ppos), idx))
        M = matrix(QQ, max(len(idx), 1), len(cols), {(r_, c_): v for c_, col in enumerate(cols) for r_, v in col.items()})
        print("    order %-22s: %d homogeneous basis elements (dots allowed), rank %d, kernel %d" % (
            name, len(cols), M.rank(), len(cols) - M.rank()))
    print("    [%.0fs]" % (time.time() - t0))


def part_z1():
    C = Ctx()
    t0 = time.time()
    d0 = C.tree(0)
    cands = candidates(C, -1, 1, True)
    idx = {}
    cols = []
    mono_flags = []
    for (I, J, sigma, n, m, a) in cands:
        el = C.mono(n, m) * C.basis(I, J, sigma)
        cols.append(vectorize(C, anticomm(C, d0, I, J, el), lambda e: C.rdeg(e) == 1, idx))
        mono_flags.append(not any(s not in I and sigma[s] in J for s in C.Wd.slots))
    M = matrix(QQ, max(len(idx), 1), len(cols), {(r_, c_): v for c_, col in enumerate(cols) for r_, v in col.items()})
    rk = M.rank()
    print("(z1) dot-free first-order candidates %d (monotone %d); dim of cycles mod r^2: %d [%.0fs]" % (
        len(cols), sum(mono_flags), len(cols) - rk, time.time() - t0))
    mon = [i for i, f in enumerate(mono_flags) if f]
    Mm = M.matrix_from_columns(mon)
    print("     monotone dot-free first-order cycles: dim %d" % (len(mon) - Mm.rank()))


if __name__ == "__main__":
    {"b": part_b, "c": part_c, "d": part_d, "z1": part_z1}[sys.argv[1]]()
