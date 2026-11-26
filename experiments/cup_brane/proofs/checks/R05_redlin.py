"""
Referee check 3b: the Reduction Lemma (Prop. 4.5) as a statement of linear algebra.

    sage -python R05_redlin.py TYPE RANK RED WORD DMAX [SEED]

w = w'.c (d'^2 = 0 is verified numerically first).  E(G) = iota_R(d') G - G iota_L(d') is linear in the
family G = (G_{J',K'}),  G_{J',K'} in e_{J'+m} A e_{K'}.  Ansatz space V: all families whose components
are integer combinations of  (dot monomial of degree <= DMAX) x (one reduced diagram for EVERY
colour-preserving bijection of the strands).  Prop. 4.5 (applied to F + G, where E(F) = 0 or not -- the
statement about G is the homogeneous one) says for a first-occurrence slot a of w':

        { G in V : E(G)_{I',K'} = 0 for all K' not containing a }  =  { G in V : E(G) = 0 }.

Both are kernels of integer matrices (rows = coefficients of the components of E(G) in the basis of A),
the first being a row-submatrix of the second; they are equal iff the ranks agree.  Ranks are computed
modulo two primes.  For slots that are not first occurrences the ranks are printed as a control.
"""
import sys, random, itertools, time
from R05_lib import *
from sage.matrix.constructor import matrix
from sage.rings.finite_rings.finite_field_constructor import GF


def colour_bijections(S, src, tgt_slots):
    """All target arrangements (in strand labels of src) compatible with the colours of tgt_slots."""
    by = defaultdict(list)
    for lab in src:
        by[S.col(lab)].append(lab)
    cols = sorted(by, key=str)
    for choice in itertools.product(*[itertools.permutations(by[cc]) for cc in cols]):
        it = {cc: iter(ch) for cc, ch in zip(cols, choice)}
        yield [next(it[S.col(lab)]) for lab in tgt_slots]


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(x) for x in sys.argv[4]); dmax = int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    rng = random.Random(seed)
    t0 = time.time()
    m = len(word); n = m - 1
    S = Setup(ct, red, word); Sp = Setup(ct, red, word[:-1])
    subsets = [frozenset(x) for k in range(n + 1) for x in itertools.combinations(range(1, n + 1), k)]
    arrR = lambda Ip: S.arr(set(Ip) | {m})
    arrL = lambda Ip: S.arr(Ip)
    dR, dL = {}, {}

    def add(d, k_, x):
        d[k_] = d[k_] + x if k_ in d else x
    for Ip in subsets:
        for t in Sp.my_terms(Ip):
            w_ = rand_reduced(Sp.arr(Ip), Sp.target_arr(Ip, t.j, t.F), rng)
            add(dR, (Ip, t.J), S.coef(t) * S.el(arrR(Ip), w_))
            add(dL, (Ip, t.J), S.coef(t) * S.el(arrL(Ip), tuple(i + 1 for i in w_)))
    succ = defaultdict(list); pred = defaultdict(list)
    for (Ip, Jp) in dR:
        succ[Ip].append(Jp); pred[Jp].append(Ip)
    for Ip in subsets:
        tot = {}
        for Jp in succ[Ip]:
            for Kp in succ[Jp]:
                add(tot, Kp, dR[Ip, Jp] * dR[Jp, Kp])
        assert all(v.is_zero() for v in tot.values()), "d'^2 != 0"

    rows = {}
    entries = defaultdict(int)
    ncol = 0
    for Jp in subsets:
        src = arrR(Jp)
        blacks = [l for l in src if l != "W"]
        dots = [S.idem(src)]
        for d_ in range(1, dmax + 1):
            for combo in itertools.combinations_with_replacement(blacks, d_):
                x = S.idem(src)
                for lab in combo:
                    x = x * S.dot(src, lab)
                dots.append(x)
        for Kp in subsets:
            for tgt in colour_bijections(S, src, arrL(Kp)):
                diag = S.el(src, rand_reduced(src, tgt, rng))
                for dm in dots:
                    g = dm * diag
                    col = ncol; ncol += 1
                    parts = [((Ip, Kp), dR[Ip, Jp] * g) for Ip in pred[Jp]] + \
                            [((Jp, Lp), -(g * dL[Kp, Lp])) for Lp in succ[Kp]]
                    for comp, val in parts:
                        for braid, coeff in val:
                            for exp, cc in coeff.iterator_exp_coeff():
                                key = (comp, braid, tuple(exp))
                                r_ = rows.setdefault(key, len(rows))
                                entries[r_, col] += int(cc)
    print("%s%d red %d word %s, dots <= %d: %d unknowns, %d equations [%.0fs]" % (
        ct[0], ct[1], red, "".join(map(str, word)), dmax, ncol, len(rows), time.time() - t0), flush=True)
    rowkeys = sorted(rows, key=lambda k_: rows[k_])
    firsts = [s for s in range(1, n + 1) if word.index(word[s - 1]) == s - 1]
    for p in (32003,):
        Mfull = matrix(GF(p), len(rows), ncol, dict(entries), sparse=True)
        rk = Mfull.rank()
        line = ["p=%d: rank(E) = %d (kernel = cocycles in V: %d)" % (p, rk, ncol - rk)]
        ok = True
        for a in range(1, n + 1):
            keep = [i for i, k_ in enumerate(rowkeys) if a not in k_[0][1]]
            rka = Mfull.matrix_from_rows(keep).rank()
            tag = "first" if a in firsts else "NOT first"
            line.append("slot %d (%s): rank restricted to a notin K' = %d%s" % (
                a, tag, rka, "" if rka == rk else "  <-- smaller"))
            if a in firsts and rka != rk:
                ok = False
        print("\n   ".join(line), flush=True)
        print("   Prop. 4.5 on the ansatz space:", "CONFIRMED" if ok else "VIOLATED", "[%.0fs]" % (time.time() - t0), flush=True)
