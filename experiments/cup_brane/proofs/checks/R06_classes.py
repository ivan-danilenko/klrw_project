"""
R06: checks the STRUCTURE of the proof of Theorem 6.2 (section 6.6), not only its statement.
For every vertex I, every term of d_Phi G and of G d is computed as an element of the algebra,
the right terms are split into main and resolved parts as in Lemma 6.5 (z_p factored out of
gamma, then z_p beta = [sigma] + r_p [sigma (a j)]), the split is checked against the actual
term, every piece is assigned to its class T0(j, F), T1(p, F), T2(p, F) by the rules of 6.6,
and each class is checked to sum to zero separately.  Also checks Lemma 6.5 directly and that
[sigma_F] equals the klrw basis braid (Definition 6.2: "a basis diagram with crossings psi only").

    sage -python R06_classes.py TYPE RANK RED WORD
"""
import sys, time
from collections import defaultdict
from R06_lib import Setup, perm_sign
from fast import braid_for, exp_for
from sample_verify import element
import os
MISASSIGN = bool(os.environ.get('R06_MISASSIGN'))   # negative control: swap T1/T2 for resolved parts


def run(ct, red, w):
    t0 = time.time()
    S = Setup(ct, red, w)
    assert S.maxmult <= 2
    twin_below = {}
    for (a, b) in S.pairs:
        twin_below[b] = a
    pair_of = {}
    for p in S.pairs:
        pair_of[p[0]] = p
        pair_of[p[1]] = p

    def Gterm(V, F, drop=None):
        """epsilon(V,F) gamma_F(V) [sigma_F]_{V -> V_F}; if drop = p, omit the factor z_p."""
        V = frozenset(V)
        a0 = S.arr(V)
        e = S.idem(a0)
        xs = {s: S.dot(a0, s) for s in range(1, S.m + 1)}
        VF = (V - {p[1] for p in F}) | {p[0] for p in F}
        rep = {p[1]: p[0] for p in F}
        coef = perm_sign([rep.get(s, s) for s in sorted(V)]) * e
        for p in F:
            coef = coef * (-S.r(S.col(p[0])) * e)
        for p in S.pairs:
            if p not in F and p[1] in V and p != drop:
                coef = coef * (xs[p[0]] - xs[p[1]])
        for p in F:
            for q in F:
                if S.interlaced_lt(p, q) and S.adj(S.col(p[0]), S.col(q[0])):
                    coef = coef * S.Q(S.col(p[0]), S.col(q[0]), xs[p[1]], xs[q[0]])
        sigma = {}
        for p in F:
            sigma[p[0]], sigma[p[1]] = p[1], p[0]
        return coef, S.bracket(V, sigma, VF), frozenset(VF)

    nclass_bad = 0
    nsplit_bad = 0
    nl65_bad = 0
    nbasis_bad = 0
    nclasses = 0
    nterms = 0
    import itertools
    for I in S.vertices():
        classes = defaultdict(lambda: S.K.zero())
        PI = [p for p in S.pairs if p[0] not in I and p[1] in I]
        # ---- left terms
        for j in sorted(I):
            sgn = (-1) ** sum(1 for i in I if i > j)
            Ij = I - {j}
            PIj = [p for p in S.pairs if p[0] not in Ij and p[1] in Ij]
            for k in range(len(PIj) + 1):
                for F in itertools.combinations(PIj, k):
                    coef, br, KK = Gterm(Ij, F)
                    el = sgn * S.Phi(I, j) * coef * br
                    inF = [p for p in F if j in p]
                    if not inF:
                        key = ("T0", j, frozenset(F))
                    else:
                        (p,) = inF
                        assert p[0] == j
                        key = ("T1", p, frozenset(set(F) - {p}))
                    classes[key] += el
                    nterms += 1
        # ---- right terms
        for k in range(len(PI) + 1):
            for F1 in itertools.combinations(PI, k):
                coef1, br1, I1 = Gterm(I, F1)
                Gt = coef1 * br1
                if True:
                    # Definition 6.2 claim: [sigma_F] is the klrw basis braid (dot-free, psi only)
                    sigma = list(range(1, S.m + 1))
                    for p in F1:
                        sigma[p[0] - 1], sigma[p[1] - 1] = p[1], p[0]
                    _, _, braid = braid_for(S.cube, I, I1, tuple(sigma))
                    nbasis_bad += not (br1 - S.K.monomial(braid)).is_zero()
                for term in S.stack.terms_from(I1):
                    J, sig, mono, sign = term
                    F2 = [(s, sig[s - 1]) for s in range(1, S.m + 1) if sig[s - 1] > s]
                    movers = set(I1) - set(J) - {b for a, b in F2}
                    (j,) = movers
                    scal = S.Dt(sign) * (S.Dt.monomial(*exp_for(S.K, mono)) if mono else S.Dt.one())
                    full = Gt * element(S.cube, S.stack, I1, term, S.cache)
                    nterms += 1
                    allF = frozenset(set(F1) | set(F2))
                    sig2 = {}
                    for p in F2:
                        sig2[p[0]], sig2[p[1]] = p[1], p[0]
                    K = frozenset(J)
                    main_br = S.bracket(I1, sig2, K)
                    if j in I and j in twin_below:
                        a = twin_below[j]
                        p = (a, j)
                        assert p not in F1
                        coefz, _, _ = Gterm(I, F1, drop=p)
                        # Lemma 6.5 directly at I1: z_p beta = [sigma] + r [sigma (a j)]
                        sig3 = dict(sig2)
                        sig3[a], sig3[j] = j, a
                        res_br = S.bracket(I1, sig3, K)
                        a1 = S.arr(I1)
                        zp = S.dot(a1, a) - S.dot(a1, j)
                        beta = element(S.cube, S.stack, I1, term, S.cache)
                        rp = S.r(S.col(a))
                        nl65_bad += not (zp * beta - scal * main_br - scal * rp * res_br).is_zero()
                        main = coefz * br1 * scal * main_br
                        resolved = coefz * br1 * scal * rp * res_br
                        nsplit_bad += not (main + resolved - full).is_zero()
                        classes[("T0", j, allF)] += -main
                        if (a in I) != MISASSIGN:
                            classes[("T1", p, allF)] += -resolved
                        else:
                            classes[("T2", p, allF)] += -resolved
                    else:
                        if j in I:
                            classes[("T0", j, allF)] += -full
                        else:
                            (p,) = [q for q in F1 if q[0] == j]
                            classes[("T2", p, frozenset(allF - {p}))] += -full
                        nsplit_bad += not (main_br * scal - element(S.cube, S.stack, I1, term, S.cache)).is_zero()
        for key, v in classes.items():
            nclasses += 1
            if not v.is_zero():
                nclass_bad += 1
                print("  class not zero:", sorted(I), key, flush=True)
    print("%s red %d word %s: %d terms, %d classes, classes != 0: %d, split errors: %d, Lemma 6.5 errors: %d, "
          "[sigma_F] != basis braid: %d  [%.1fs]" % ("".join(map(str, ct)), red, "".join(map(str, w)), nterms, nclasses,
                                                    nclass_bad, nsplit_bad, nl65_bad, nbasis_bad, time.time() - t0), flush=True)
    return nclass_bad + nsplit_bad + nl65_bad + nbasis_bad




def batch(ct, minlen, maxlen, part, nparts):
    import itertools
    from collections import Counter
    cols = list(range(1, ct[1] + 1))
    W = [w for L in range(minlen, maxlen + 1) for w in itertools.product(cols, repeat=L)
         if max(Counter(w).values()) <= 2][part::nparts]
    bad = 0
    n = 0
    for w in W:
        for red in cols:
            bad += bool(run(ct, red, w))
            n += 1
    print("RESULT classes %s len %d..%d part %d/%d: %d cases, %d with errors" % (
        "".join(map(str, ct)), minlen, maxlen, part, nparts, n, bad), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "all":
        ct = [sys.argv[2], int(sys.argv[3])]
        part, nparts = map(int, sys.argv[6].split("/"))
        batch(ct, int(sys.argv[4]), int(sys.argv[5]), part, nparts)
    else:
        ct = [sys.argv[1], int(sys.argv[2])]
        run(ct, int(sys.argv[3]), tuple(int(c) for c in sys.argv[4]))
