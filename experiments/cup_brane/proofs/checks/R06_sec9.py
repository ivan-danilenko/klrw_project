"""
R06: Proposition 9.2 (Koszul subcomplexes K(t) of C_Phi): for labelled states t and all I, j in I,
    D(t -> std I) Phi_j(I) = f_j(t) D(t -> std(I - j)),
    f_j(t) = prod{ q_{jy} : y in {W, 1..j-1}, j to the left of y in t }.
Also: Delta (product over same-colour position pairs of (x_p - x_q)^2) commutes with generators.

    sage -python R06_sec9.py
"""
import random, itertools
from R06_lib import Setup

rng = random.Random(9)
cases = [(["A", 2], 1, (1, 2, 1, 2)), (["A", 2], 2, (1, 2, 2, 1)), (["A", 3], 1, (1, 2, 3, 1, 2)),
         (["D", 4], 2, (2, 1, 3, 2, 4)), (["A", 2], 1, (1, 1, 2, 1))]
total = bad = 0
for ct, red, w in cases:
    S = Setup(ct, red, w)
    labels = list(range(1, S.m + 1)) + ["W"]

    def q(t, a, b):
        if b == "W":
            return S.Dt("u_%d" % red) * S.dot(t, a) if S.col(a) == red else S.idem(t)
        ca, cb = S.col(a), S.col(b)
        if ca == cb:
            return S.r(ca) ** 2 * S.idem(t)
        if S.adj(ca, cb):
            return S.t(ca, cb) * S.dot(t, a) + S.t(cb, ca) * S.dot(t, b)
        return S.idem(t)

    nb = nt = 0
    for trial in range(6):
        t = labels[:]
        rng.shuffle(t)
        pos = {lab: i for i, lab in enumerate(t)}
        for I in S.vertices():
            DI = S.D(t, S.arr(I))
            for j in I:
                f = S.idem(t)
                for y in ["W"] + list(range(1, j)):
                    if pos[j] < pos[y]:
                        f = f * q(t, j, y)
                lhs = DI * S.Phi(I, j)
                rhs = f * S.D(t, S.arr(set(I) - {j}))
                nt += 1
                nb += not (lhs - rhs).is_zero()
    # Delta central (commutes with every crossing and dot at random states)
    ndelta = 0
    for trial in range(10):
        s = labels[:]
        rng.shuffle(s)
        i = rng.randint(1, len(s) - 1)
        def Delta(arr):
            d = S.idem(arr)
            for p1, p2 in itertools.combinations(range(1, len(arr) + 1), 2):
                if "W" not in (arr[p1 - 1], arr[p2 - 1]) and S.col(arr[p1 - 1]) == S.col(arr[p2 - 1]):
                    d = d * (S.dotpos(arr, p1) - S.dotpos(arr, p2)) ** 2
            return d
        s2 = list(s); s2[i - 1], s2[i] = s2[i], s2[i - 1]
        ndelta += not (Delta(s) * S.psi(s, i) - S.psi(s, i) * Delta(s2)).is_zero()
    print("%s red %d word %s: Prop 9.2 identity %d/%d failures; Delta central failures %d/10" % (
        "".join(map(str, ct)), red, "".join(map(str, w)), nb, nt, ndelta), flush=True)
    total += nt
    bad += nb + ndelta
print("RESULT sec9: %d identities, %d failures" % (total, bad))
