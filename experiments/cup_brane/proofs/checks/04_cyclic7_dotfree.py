"""
Theorem 9.1 of proofs/04_first_order.md: completions of the tree formula of the cyclic word
1221122 (A_2, red strand of colour 2).  "Dot-free" and "basis element" always refer to the
klrw basis braid of a bijection (fast.braid_for), dots at the left end.

    sage -python proofs/checks/04_cyclic7_dotfree.py

(1) number of dot-free homogeneous basis elements of homological degree -1 by r-degree
    (pure-Python enumeration, degree_model.Model.candidates);
(2) the r-degree-2, 3, 4 parts of d_T^2 (exact, klrw algebra over Z), d_T = tree formula;
(3) Z2 = dot-free r-degree-2 elements kappa with [delta_0, kappa] = 0 mod r^3, and the
    r^3-parts of [delta_0 + delta_1, kappa] = (delta_0+delta_1) kappa + kappa (delta_0+delta_1).
    REVISED after R04_referee.md (G0): the first version took the left factors only from the
    vertices I u {x}, which drops every product delta_1 * kappa (a component of delta_1 ending
    in I starts at I - a + {j, b}).  Now all left factors are taken.  Result: rank 4 -> 4,
    the obstruction IS removable; the unique solution kappa* is printed;
(4) the r^4 step: (d_T + kappa*)^2 has r-degree >= 4; the 7 dot-free r^4 candidates absorb it;
    d_F := d_T + kappa* + eps_4 is built and d_F^2 = 0 is checked exactly;
(5) independent data: the referee's d_F (checks/R04_data/cyclic7_1221122_W2_dotfree.json) is
    rebuilt with this code from the JSON alone and compared with (4); d^2 = 0 is checked;
(6) gauge: with d_X = d_T + X (our dotted completion, checks/04_data/cyclic7_1221122_W2_tree.json)
    and the referee's h (checks/R04_data/cyclic7_1221122_W2_gauge_h.json), h^2 = 0 and
    (1 + h) d_X (1 - h) = d_F are checked exactly.
"""
import sys, os, json, importlib, time
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L = importlib.import_module("04_lib")
F = importlib.import_module("04_forced")
from sage.all import matrix, QQ, vector

WORD = (1, 2, 2, 1, 1, 2, 2)
RED = 2


def rdeg_of(mono):
    return sum(p for n, p in mono if n.startswith("r"))


def add_into(C, tot, key, el):
    tot[key] = tot.get(key, C.K.zero()) + el


def bracket(C, V, maps, comp):
    """sum over maps m of (m comp + comp m), comp = dict (I, J) -> element."""
    tot = {}
    for (I, J), el in comp.items():
        for mp in maps:
            for S in V:                       # ALL left factors m_{S,I}
                if I in mp[S]:
                    add_into(C, tot, (S, J), mp[S][I] * el)
            for Jp, el2 in mp[J].items():
                add_into(C, tot, (I, Jp), el * el2)
    return tot


def as_map(C, V, comps):
    mp = {I: defaultdict(lambda: C.K.zero()) for I in V}
    for (I, J), el in comps.items():
        mp[I][J] += el
    return {I: dict(row) for I, row in mp.items()}


def map_sum(C, V, *maps):
    res = {I: defaultdict(lambda: C.K.zero()) for I in V}
    for mp in maps:
        for I, row in mp.items():
            for J, el in row.items():
                res[I][J] += el
    return {I: dict(row) for I, row in res.items()}


def square(C, mp):
    sq = L.compose(C, mp, mp)
    return {(I, K): v for I, row in sq.items() for K, v in row.items() if not v.is_zero()}


def element(C, I, J, sigma, monodict, coeff):
    mon = C.D.one()
    for nm, p in monodict.items():
        mon *= C.D(nm) ** p
    return (C.D(coeff) * mon) * C.basis(frozenset(I), frozenset(J), tuple(sigma))


def map_from_terms(C, V, terms):
    comps = defaultdict(lambda: C.K.zero())
    for I, J, sigma, monodict, coeff in terms:
        comps[frozenset(I), frozenset(J)] += element(C, I, J, sigma, monodict, coeff)
    return as_map(C, V, comps)


def maps_equal(C, a, b):
    for I in set(a) | set(b):
        for J in set(a.get(I, {})) | set(b.get(I, {})):
            if a.get(I, {}).get(J, C.K.zero()) != b.get(I, {}).get(J, C.K.zero()):
                return False
    return True


if __name__ == "__main__":
    t0 = time.time()
    C = L.Ctx(["A", 2], RED, WORD)
    V = list(L.subsets(C.m))
    cands = C.model.candidates()
    byr = Counter(rdeg_of(mono) for I, J, s, mono in cands)
    print("(1) dot-free homogeneous candidates (hom. degree -1) by r-degree:", sorted(byr.items()))
    d0 = {I: C.delta0(I) for I in V}
    d1 = {I: C.delta_n(I, 1) for I in V}
    d2 = {I: C.delta_n(I, 2) for I in V}
    dT = map_sum(C, V, d0, d1, d2)
    sq = square(C, dT)
    cnt = {k: len(F.vectorize(C, sq, k, {})) for k in range(0, 7)}
    print("(2) coefficients of d_T^2 by r-degree:", cnt)
    idx = {}
    obs = F.vectorize(C, sq, 3, idx)
    # (3)
    ck = [(I, J, s, mono) for I, J, s, mono in cands if rdeg_of(mono) == 2]
    idk = {}
    cols = [F.vectorize(C, F.commutator(C, d0, I, J, C.mono(mono) * C.basis(I, J, s), +1), 2, idk) for I, J, s, mono in ck]
    K = F.mat(cols, max(len(idk), 1)).right_kernel().basis()
    print("(3) r-degree 2: %d dot-free candidates; kernel of [delta_0, -] mod r^3: dim %d" % (len(ck), len(K)))
    kaps = []
    for v in K:
        comp = defaultdict(lambda: C.K.zero())
        for i, c in v.dict().items():
            assert c == int(c)
            I, J, s, mono = ck[i]
            comp[I, J] += C.D(int(c)) * (C.mono(mono) * C.basis(I, J, s))
        kaps.append(dict(comp))
    vecs = [F.vectorize(C, bracket(C, V, (d0, d1), kap), 3, idx) for kap in kaps]
    n = len(idx)
    Mk = F.mat(vecs, n)
    ob = vector(QQ, n, obs)
    r1 = Mk.rank(); r2 = Mk.augment(ob.column()).rank()
    print("    r^3-parts of [delta_0 + delta_1, kappa], kappa in Z2: rank %d; with (d_T^2)_3: rank %d -> %s" % (
        r1, r2, "NOT removable" if r2 > r1 else "removable"))
    sol = Mk.solve_right(-ob)
    print("    kappa* = combination", list(sol), "of the kernel basis (unique: rank = dim Z2 = %d)" % len(K))
    kstar = defaultdict(lambda: C.K.zero())
    for c, kap in zip(sol, kaps):
        for key, el in kap.items():
            kstar[key] += C.D(int(c)) * el
    kstar = {k: v for k, v in kstar.items() if not v.is_zero()}
    for (I, J), el in sorted(kstar.items(), key=lambda kv: (sorted(kv[0][0]), sorted(kv[0][1]))):
        for b, e, c in C.terms(el):
            sig = C.bijection(I, J, b)
            print("      %+d %s  %s -> %s  sigma %s" % (c, C.fmt_exp(e), sorted(I), sorted(J),
                                                    {s: sig[s - 1] for s in range(1, C.m + 1) if sig[s - 1] != s}))
    # (4)
    d1k = map_sum(C, V, dT, as_map(C, V, kstar))
    sq1 = square(C, d1k)
    print("(4) coefficients of (d_T + kappa*)^2 by r-degree:", {k: len(F.vectorize(C, sq1, k, {})) for k in range(0, 7)})
    c4 = [(I, J, s, mono) for I, J, s, mono in cands if rdeg_of(mono) == 4]
    id4 = {}
    ob4 = F.vectorize(C, sq1, 4, id4)
    cols4 = [F.vectorize(C, F.commutator(C, d0, I, J, C.mono(mono) * C.basis(I, J, s), +1), 4, id4) for I, J, s, mono in c4]
    M4 = F.mat(cols4, len(id4))
    v4 = vector(QQ, len(id4), ob4)
    print("    r^4: %d dot-free candidates, rank %d, with obstruction %d" % (len(c4), M4.rank(), M4.augment(v4.column()).rank()))
    s4 = M4.solve_right(-v4)
    eps4 = defaultdict(lambda: C.K.zero())
    for c, (I, J, s, mono) in zip(s4, c4):
        if c:
            eps4[I, J] += C.D(int(c)) * (C.mono(mono) * C.basis(I, J, s))
            print("      eps_4: %+d %s  %s -> %s  sigma %s" % (int(c), "*".join("%s^%d" % x for x in mono), sorted(I), sorted(J),
                                                             {a: s[a - 1] for a in range(1, C.m + 1) if s[a - 1] != a}))
    dF = map_sum(C, V, d1k, as_map(C, V, dict(eps4)))
    print("    d_F = d_T + kappa* + eps_4:  d_F^2 = 0 exactly: %s;  number of terms %d" % (
        len(square(C, dF)) == 0, sum(len(C.terms(el)) for row in dF.values() for el in row.values())))
    # (5)
    ref = json.load(open(os.path.join(HERE, "R04_data", "cyclic7_1221122_W2_dotfree.json")))
    dR = map_from_terms(C, V, ref["terms"])
    print("(5) referee's d_F (%d terms, from JSON): d^2 = 0: %s;  equal to our d_F: %s" % (
        len(ref["terms"]), len(square(C, dR)) == 0, maps_equal(C, dR, dF)))
    # (6)
    mine = json.load(open(os.path.join(HERE, "04_data", "cyclic7_1221122_W2_tree.json")))
    tX = [(I, J, s, dict([tuple(x) for x in pars] + [tuple(x) for x in dots]), c) for I, J, s, pars, dots, c in mine["terms"]]
    dX = map_from_terms(C, V, tX)
    hj = json.load(open(os.path.join(HERE, "R04_data", "cyclic7_1221122_W2_gauge_h.json")))
    h = map_from_terms(C, V, hj["terms"])
    hneg = map_from_terms(C, V, [(I, J, s, m, -c) for I, J, s, m, c in hj["terms"]])
    one = {K_: {K_: C.K.idempotent(C.cube.state(C.eps(K_)))} for K_ in V}
    g = map_sum(C, V, one, h); gi = map_sum(C, V, one, hneg)
    hh = L.compose(C, h, h)
    conj = L.compose(C, L.compose(C, g, dX), gi)
    conj = {I: {J: el for J, el in row.items() if not el.is_zero()} for I, row in conj.items()}
    print("(6) d_X (ours, %d terms): d_X^2 = 0: %s;  h^2 = 0: %s;  (1+h) d_X (1-h) = d_F: %s" % (
        len(tX), len(square(C, dX)) == 0, all(el.is_zero() for row in hh.values() for el in row.values()),
        maps_equal(C, conj, dF)))
    # (7) monotonicity and comparison with the tree terms
    def monotone(mp):
        for I, row in mp.items():
            for J, el in row.items():
                for b, e, c in C.terms(el):
                    sig = C.bijection(I, J, b)
                    if any(s not in I and sig[s - 1] in J for s in range(1, C.m + 1)):
                        return False
        return True
    tree_keys = set((I, J, sig) for (I, J, sig, mono) in C.st.all_terms())
    extra = [(I, J, C.bijection(I, J, b)) for I, row in map_sum(C, V, dF, {I: {J: -el for J, el in row.items()} for I, row in dT.items()}).items()
             for J, el in row.items() for b, e, c in C.terms(el)]
    print("(7) monotone (no strand crosses W from left to right): d_F %s, d_X %s, h %s;  extra terms of d_F: %d, "
          "of which with the bijection of a tree term: %d" % (monotone(dF), monotone(dX), monotone(h), len(extra),
                                                             sum(1 for x in extra if x in tree_keys)))
    print("[%.0fs]" % (time.time() - t0))
