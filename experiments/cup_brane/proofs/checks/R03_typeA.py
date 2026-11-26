"""
Referee test of Lemma 3.1, Lemma 3.2, Theorem A, Prop. 3.3(a), Cor. 3.4, Prop. 2.11(b),(c) (type A).

  python3 R03_typeA.py [B]      (B = box size, default 6: all diagrams in the B x B box)

Model built here: k = B, n = 2B, box (r, s) (1-based) has colour k + s - r, word = row-major.
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *


def partitions_in_box(rows, cols):
    def rec(i, maxv):
        if i == rows:
            yield ()
            return
        for v in range(maxv, -1, -1):
            for rest in rec(i + 1, v):
                yield (v,) + rest
    for p in rec(0, cols):
        yield p


def boxes(lam):
    return [(r + 1, s + 1) for r in range(len(lam)) for s in range(lam[r])]


def word_rowmajor(lam, k):
    return [k + s - r for (r, s) in boxes(lam)]


def word_colmajor(lam, k):
    bx = sorted(boxes(lam), key=lambda b: (b[1], b[0]))
    return [k + s - r for (r, s) in bx]


def contains(lam, mu):
    return all((lam[i] if i < len(lam) else 0) >= mu[i] for i in range(len(mu)))


def main(B):
    k, n = B, 2 * B
    E = edges("A", n - 1); adj = make_adj(E)
    t0 = time.time()
    st = dict(diagrams=0, cyclic=0, crit_disagree=0, pat_disagree=0, heap_disagree=0, not_chain=0,
              lemma32_fail=0, visited_max=0)
    weight_of = {}
    for lam in partitions_in_box(k, n - k):
        lam = tuple(x for x in lam if x > 0)
        w = word_rowmajor(lam, k)
        ok, mu = is_chain_word(w, E, n - 1, k)
        if not ok:
            st["not_chain"] += 1
        weight_of[lam] = mu
        st["diagrams"] += 1
        T = Tables(w, adj)
        res, vis = conn_search_fast(T)
        st["visited_max"] = max(st["visited_max"], vis)
        cyc = res is not None
        crit = contains(lam, (4, 4, 4, 3))
        if cyc:
            st["cyclic"] += 1
        if cyc != crit:
            st["crit_disagree"] += 1
            print("THEOREM A DISAGREE", lam, cyc, res)
        if has_pattern(w, E) != cyc:
            st["pat_disagree"] += 1
            print("PATTERN DISAGREE", lam)
        w2 = word_colmajor(lam, k)
        r2, _ = conn_search_fast(Tables(w2, adj))
        if (r2 is not None) != cyc:
            st["heap_disagree"] += 1
            print("ROW/COL-MAJOR DISAGREE", lam)
        if not crit and max([w.count(c) for c in set(w)] + [0]) > 3:
            st["lemma32_fail"] += 1
            print("LEMMA 3.2 FAIL", lam)
    print("Theorem A, %dx%d box:" % (B, B), st, "(%.1fs)" % (time.time() - t0), flush=True)

    # Stembridge/Proctor input, checked here: chain words to each weight form one commutation class,
    # the row-major words of the diagrams hit every weight once, and weight order = reverse inclusion.
    t0 = time.time()
    heaps, covers, order = all_weights_with_heaps(E, n - 1, k)
    single = all(len(h) == 1 for h in heaps.values())
    hit = {}
    match = 0
    for lam, mu in weight_of.items():
        hit.setdefault(mu, []).append(lam)
        if heaps.get(mu) == {heap_key(tuple(word_rowmajor(lam, k)), E)}:
            match += 1
    bij = len(hit) == len(heaps) and all(len(v) == 1 for v in hit.values())
    # weight order: mu <= nu iff content(mu) >= content(nu) componentwise (content = omega_k - mu in roots)
    cont = {lam: [word_rowmajor(lam, k).count(c) for c in range(1, n)] for lam in weight_of}
    bad = 0
    L = list(weight_of)
    for a in L:
        for b in L:
            inc = contains(a, b)
            wo = all(x >= y for x, y in zip(cont[a], cont[b]))
            if inc != wo:
                bad += 1
    print("A_%d omega_%d: %d weights; every weight has ONE heap of chain words: %s; row-major words hit"
          " every weight exactly once: %s; heap of chains = heap of row-major word: %d/%d;"
          " weight order vs inclusion mismatches: %d  (%.1fs)" % (
              n - 1, k, len(heaps), single, bij, match, len(weight_of), bad, time.time() - t0), flush=True)

    # Prop 3.3(a) and the census of minimal cyclic configurations (brute force over all configurations)
    def minimal_all(lam):
        w = word_rowmajor(lam, k)
        T = Tables(w, adj)
        allc = T.brute(first_only=False)
        by = {}
        for j, F in allc:
            by.setdefault(j, []).append(frozenset(F))
        mins = []
        for j, Ls in by.items():
            for F in Ls:
                if not any(G < F for G in Ls):
                    mins.append((j, F))
        bx = boxes(lam)
        return [(bx[j], sorted((bx[a], bx[b]) for a, b in F)) for j, F in mins], len(allc)
    for lam in [(4, 4, 4, 3), (5, 4, 4, 3), (4, 4, 4, 3, 1), (5, 5, 4, 3), (4, 4, 4, 4), (5, 5, 5, 5)]:
        if len(lam) > k or max(lam) > n - k:
            continue
        t1 = time.time()
        mins, ncyc = minimal_all(lam)
        if len(mins) <= 2:
            print("  %s: %d cyclic configurations, %d minimal: %s (%.1fs)" % (lam, ncyc, len(mins), mins, time.time() - t1))
        else:
            ls = sorted(len(F) for _, F in mins)
            print("  %s: %d cyclic configurations, %d minimal, sizes %s (%.1fs)" % (lam, ncyc, len(mins), ls, time.time() - t1))
        sys.stdout.flush()

    # Prop 2.11(b): 3 x 5 rectangle (sl_8, omega_3), column-major word 321432543654765
    k3, n3 = 3, 8
    E3 = edges("A", 7); adj3 = make_adj(E3)
    w = word_colmajor((5, 5, 5), k3)
    print("sl_8 omega_3 rectangle, column-major word", "".join(map(str, w)),
          " cyclic:", conn_search_fast(Tables(w, adj3))[0],
          " ignoring blocks:", conn_search_fast(Tables(w, adj3), ignore_blocks=True)[0],
          " analyse(slot8; (4,12),(1,9),(7,15)):", analyse(w, adj3, 7, [(3, 11), (0, 8), (6, 14)]))

    # Prop 2.11(c): sl_10 omega_5: cyclic vs cyclic-ignoring-blocks
    k5, n5 = 5, 10
    E5 = edges("A", 9); adj5 = make_adj(E5)
    nc = nib = 0
    t1 = time.time()
    for lam in partitions_in_box(5, 5):
        lam = tuple(x for x in lam if x > 0)
        w = word_rowmajor(lam, k5)
        T = Tables(w, adj5)
        if conn_search_fast(T)[0] is not None:
            nc += 1
        if conn_search_fast(T, ignore_blocks=True)[0] is not None:
            nib += 1
    print("sl_10 omega_5 (5x5 box, 252 diagrams): cyclic %d, cyclic ignoring blocks %d (%.1fs)" % (nc, nib, time.time() - t1))
    # Cor 3.4 counts
    for (kk, nn) in [(4, 8), (4, 9), (4, 10), (5, 10)]:
        cnt = sum(1 for lam in partitions_in_box(kk, nn - kk) if contains(lam, (4, 4, 4, 3)))
        print("  A_%d omega_%d: %d diagrams contain (4,4,4,3)" % (nn - 1, kk, cnt))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
