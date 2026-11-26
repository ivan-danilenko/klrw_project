"""
Referee re-derivation of Theorem E (E_6 omega_1, omega_6; E_7 omega_7) and Prop. 2.11(c) for E_6.

  python3 R03_typeE.py

Weights by BFS of chain steps from omega_k; for each weight the set of heaps of ALL chain words
(DP), which must be a single commutation class; its canonical linear extension is the stack word.
Cyclicity by (1) brute force over all configurations, (2) exhaustive connected growth,
(3) the T/B/N detector.  Then: number of cyclic weights, minimal ones in the chain-reachability
order, mu_c, multiplicities, and all inclusion-minimal cyclic configurations of I_c with their
pattern type.
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *


def run(n, k):
    t0 = time.time()
    E = edges("E", n); adj = make_adj(E)
    heaps, covers, order = all_weights_with_heaps(E, n, k)
    single = all(len(h) == 1 for h in heaps.values())
    word = {mu: list(next(iter(h))) for mu, h in heaps.items()}
    cyc = {}
    agree = True
    for mu in order:
        w = word[mu]
        T = Tables(w, adj)
        b = T.brute() is not None
        c = conn_search_fast(T)[0] is not None
        p = has_pattern(w, E)
        if not (b == c == p):
            agree = False
            print("DISAGREE", mu, b, c, p)
        cyc[mu] = b
    ncyc = sum(cyc.values())
    lam = order[0]
    # reachability order: nu reachable from mu  <=> I_mu subset I_nu
    reach = {mu: reachable(covers, mu) for mu in order}
    cyc_list = [mu for mu in order if cyc[mu]]
    mins = [mu for mu in cyc_list if not any(nu != mu and mu in reach[nu] for nu in cyc_list)]
    upset_ok = all(cyc[nu] == any(nu in reach[m] for m in mins) for nu in order)
    # weight order (mu <= nu iff nu - mu in N-span of simple roots) vs reachability
    def content(mu):
        w = word[mu]
        return [w.count(c) for c in range(1, n + 1)]
    ordbad = sum(1 for a in order for b in order
                 if (b in reach[a]) != all(x >= y for x, y in zip(content(b), content(a))))
    noncyc_mult = max(max([word[mu].count(c) for c in set(word[mu])] + [0]) for mu in order if not cyc[mu])
    print("E_%d omega_%d: %d weights (one heap per weight: %s); cyclic %d; three methods agree: %s;"
          " cyclic set = up-set of minimal %s; reach-vs-weight-order mismatches %d; max multiplicity of"
          " non-cyclic %d  (%.1fs)" % (n, k, len(order), single, ncyc, agree, upset_ok, ordbad, noncyc_mult,
                                      time.time() - t0))
    for m in mins:
        w = word[m]
        print("   minimal cyclic ideal: %d slots, word %s, mu_c (fund. coords) %s, omega-mu_c = %s" % (
            len(w), "".join(map(str, w)), m, content(m)))
        T = Tables(w, adj)
        allc = T.brute(first_only=False)
        by = {}
        for j, F in allc:
            by.setdefault(j, []).append(frozenset(F))
        mins_c = [(j, tuple(sorted(F))) for j, L in by.items() for F in L if not any(G < F for G in L)]
        print("   I_c: %d cyclic configurations, %d inclusion-minimal:" % (len(allc), len(mins_c)))
        for j, F in sorted(mins_c):
            at = attached_type(w, E, j, F) if len(F) == 3 else set()
            print("      mover %d (c%d), pairs %s -> %s" % (j + 1, w[j], [(a + 1, b + 1, "c%d" % w[a]) for a, b in F], at or "-"))
        # the three-pair cyclic configurations of I_c
        three = [(j, F) for j, F in allc if len(F) == 3]
        print("   I_c: three-pair cyclic configurations: %d, types %s" % (
            len(three), sorted(tuple(sorted(attached_type(w, E, j, F))) for j, F in three)))
    return order, word, cyc, reach, E, adj


if __name__ == "__main__":
    for n, k in [(6, 1), (6, 6), (7, 7)]:
        order, word, cyc, reach, E, adj = run(n, k)
        if (n, k) == (6, 1):
            # 13-slot ideal(s), and cyclic ignoring blocks
            for mu in order:
                if len(word[mu]) == 13:
                    print("   13-slot ideal %s word %s: cyclic %s" % (mu, "".join(map(str, word[mu])), cyc[mu]))
            ib = sorted(len(word[mu]) for mu in order if not cyc[mu] and
                        conn_search_fast(Tables(word[mu], adj), ignore_blocks=True)[0] is not None)
            print("   E_6 omega_1: sizes of ideals cyclic ignoring blocks but not cyclic:", ib)
        sys.stdout.flush()
