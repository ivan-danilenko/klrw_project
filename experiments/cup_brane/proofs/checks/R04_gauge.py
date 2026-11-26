"""
Referee checks of section 8 of proofs/04_first_order.md (own enumeration, own reduced words).

    sage -python proofs/checks/R04_gauge.py TYPE RANK RED WORD

Prints, for first order (r-degree exactly 1, dots allowed):
  #V^{-1}_1, #V^0_1 (all / monotone), dim Z_1, dim B_1 (all / monotone), whether the tree
  formula's delta_1 is monotone, and the number of r-free t-dependent homogeneous elements of
  homological degree 0 (Remark 8.7).
"""
import sys, os, itertools, time
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from R04_alg import AW, full_d, mcompose, madd  # noqa: E402
from R04_comb import W  # noqa: E402
from sage.all import matrix, QQ  # noqa: E402


def enum(C, hdeg, rdeg, need_t=False):
    Wd = C.Wd
    bij = list(Wd.bijections())
    V = list(Wd.subsets())
    out = []
    for I in V:
        for J in V:
            if len(J) != len(I) + hdeg:
                continue
            for sigma in bij:
                if hdeg == 0 and I == J and all(sigma[s] == s for s in Wd.slots):
                    continue
                for (n, m, l, a) in Wd.solve(Wd.defect(I, J, sigma)):
                    if l or sum(n.values()) != rdeg:
                        continue
                    if need_t and not m:
                        continue
                    mono = C.D.one()
                    for c, p in n.items():
                        mono *= C.r(c) ** p
                    for (x, y), p in m.items():
                        mono *= C.t(x, y) ** p
                    monotone = not any(s not in I and sigma[s] in J for s in Wd.slots)
                    for dm in dots(C, I, a):
                        out.append((I, J, sigma, mono * dm, monotone))
    return out


def dots(C, I, a):
    labs = [l for l in C.Wd.labels(I) if l != "W"]
    per = []
    for c, k in sorted(a.items()):
        nstr = sum(1 for l in labs if C.Wd.c(l) == c)
        vs = [C.D("x_%d_%d" % (c, i)) for i in range(1, nstr + 1)]
        lst = []
        for comb in itertools.combinations_with_replacement(vs, k):
            e = C.D.one()
            for f in comb:
                e *= f
            lst.append(e)
        per.append(lst)
    res = []
    for combo in itertools.product(*per):
        e = C.D.one()
        for f in combo:
            e *= f
        res.append(e)
    return res


def vec(C, mp, k, idx):
    v = {}
    for I, row in mp.items():
        for K, x in row.items():
            for braid, coeff in C.rpart_eq(x, k):
                for e, c in coeff.dict().items():
                    key = (I, K, braid, e)
                    if key not in idx:
                        idx[key] = len(idx)
                    v[idx[key]] = v.get(idx[key], 0) + int(c)
    return {a: b for a, b in v.items() if b}


def rank(cols, n):
    if not cols:
        return 0
    return matrix(QQ, max(n, 1), len(cols), {(r_, c_): v for c_, col in enumerate(cols) for r_, v in col.items()}).rank()


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    word = tuple(int(c) for c in sys.argv[4])
    t0 = time.time()
    C = AW(ct, red, word)
    d0 = full_d(C, 0)
    V1 = enum(C, -1, 1)
    V0 = enum(C, 0, 1)
    idx = {}
    zc = []
    for (I, J, s, cf, mon) in V1:
        eta = {I: {J: cf * C.el(I, J, s)}}
        zc.append(vec(C, madd(C, (1, mcompose(C, d0, eta)), (1, mcompose(C, eta, d0))), 1, idx))
    rz = rank(zc, len(idx))
    monidx = [i for i, t in enumerate(V1) if t[4]]
    rzm = rank([zc[i] for i in monidx], len(idx))
    idx2 = {}
    bc = []
    for (I, J, s, cf, mon) in V0:
        h = {I: {J: cf * C.el(I, J, s)}}
        bc.append(vec(C, madd(C, (1, mcompose(C, h, d0)), (-1, mcompose(C, d0, h))), 1, idx2))
    rb = rank(bc, len(idx2))
    mon0 = [i for i, t in enumerate(V0) if t[4]]
    rbm = rank([bc[i] for i in mon0], len(idx2))
    # are the monotone gauge directions themselves monotone? (products of monotone maps are)
    Vt = enum(C, 0, 0, need_t=True)
    print("%s red %d word %s: #V^-1_1 = %d (monotone %d), #V^0_1 = %d (monotone %d); dim Z1 = %d, dim B1 = %d; "
          "monotone: dim Z1^mon = %d, dim B1^mon = %d; r-free t-dependent hom-degree-0 elements: %d [%.0fs]" % (
              "".join(map(str, ct)), red, "".join(map(str, word)), len(V1), len(monidx), len(V0), len(mon0),
              len(V1) - rz, rb, len(monidx) - rzm, rbm, len(Vt), time.time() - t0))
