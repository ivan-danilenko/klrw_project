"""
Example 8.3 and Remark 8.7 of proofs/04_first_order.md: gauge freedom.

    sage -python proofs/checks/04_gauge_example.py [WORD] [RED] [ORDER]   default 1212, red 2, ORDER 1

ORDER 1: h runs over the homogeneous elements of r-degree 1 (first-order gauge, Example 8.3);
ORDER 0: h runs over the homogeneous elements of r-degree 0 containing some t (Remark 8.7:
         an r-free, t-dependent gauge that keeps the geometric part delta_0).

Takes the tree-formula differential d (a differential for this word: checked), the first-order
homogeneous elements h of homological degree 0 (04_degree enumeration), g = 1 + h, and checks
  * d^2 = 0;
  * d' = g d g^{-1} satisfies d'^2 = 0, d' - delta_0 in (r)A, and d' - d is NOT 0 mod (r)^2.
So a differential satisfying hypothesis (H1) is not determined mod (r)^2: its first-order
part is only determined up to [h, delta_0].
"""
import sys, os, importlib
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
L = importlib.import_module("04_lib")
F = importlib.import_module("04_forced")


def mapsum(C, *maps):
    res = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for coeff, mp in maps:
        for I, row in mp.items():
            for J, el in row.items():
                res[I][J] += C.D(coeff) * el
    return res


def is_zero(C, mp, maxr=None):
    for I, row in mp.items():
        for J, el in row.items():
            e = el if maxr is None else C.max_r_part(el, maxr)
            if not e.is_zero():
                return False
    return True


if __name__ == "__main__":
    word = tuple(int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "1212"))
    red = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    order = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    C = L.Ctx(["A", 2], red, word)
    V = list(L.subsets(C.m))
    d = {}
    for I in V:
        row = defaultdict(lambda: C.K.zero())
        for n in range(0, C.m):
            for J, el in (C.delta0(I) if n == 0 else C.delta_n(I, n)).items():
                row[J] += el
        d[I] = dict(row)
    print("word", "".join(map(str, word)), "red", red, ": tree formula d^2 = 0:", is_zero(C, L.compose(C, d, d)))
    H = F.space(C, 0, order, need_t=(order == 0))
    print("homogeneous elements of homological degree 0 and r-degree %d%s:" % (order, " with t" if order == 0 else ""), len(H))
    for I, J, el, desc in H:
        sigma = desc[0]
        print("   h: %s -> %s, sigma %s, element %s" % (sorted(I), sorted(J), {s: sigma[s - 1] for s in range(1, C.m + 1) if sigma[s - 1] != s}, el))
        h = {I: {J: el}}
        hinv = {I: {J: -el}}          # h^2 = 0 (one component between two distinct vertices)
        assert I != J and is_zero(C, L.compose(C, h, h))
        one = {K: {K: C.K.idempotent(C.cube.state(C.eps(K)))} for K in V}
        g = mapsum(C, (1, one), (1, h))
        gi = mapsum(C, (1, one), (1, hinv))
        dp = L.compose(C, L.compose(C, g, d), gi)
        sq = L.compose(C, dp, dp)
        diff = mapsum(C, (1, dp), (-1, d))
        d0 = {K: C.delta0(K) for K in V}
        print("      d' = g d g^-1:  d'^2 = 0: %s;  d' - delta_0 = 0 mod r: %s;  d' = d mod r^2: %s" % (
            is_zero(C, sq), is_zero(C, mapsum(C, (1, dp), (-1, d0)), 0), is_zero(C, diff, 1)))
        if order == 0:
            params = [p for p in range(len(C.names)) if not C.names[p].startswith("x_")]
            geo = all(any(e[p] for p in params)
                      for row in diff.values() for el in row.values() for _, e, c in C.terms(el))
            print("      every term of d' - d contains a parameter (geometric part unchanged): %s" % geo)
        for I2, row in diff.items():
            for J2, e2 in row.items():
                e1 = C.max_r_part(e2, 1)
                if not e1.is_zero():
                    print("        first-order change %s -> %s: %s" % (sorted(I2), sorted(J2), e1))
