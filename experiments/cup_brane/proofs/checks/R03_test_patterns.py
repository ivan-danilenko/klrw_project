"""
Referee test of Theorem 2.8 (pattern theorem), Cor. 2.10, Lemma 2.3(a) shape, Prop. 2.11(a).

  python3 R03_test_patterns.py [quick]

For every word in the listed families:
  brute  = brute force over ALL configurations (R03_core.Tables.brute), definition only, no bound
  conn   = exhaustive growth of connected unblocked configurations (R03_core.connected_search)
  pat    = literal T/B/N detector written from section 1.5 (R03_core.patterns)
and also:
  * every cyclic configuration with exactly 3 pairs is attached to a pattern occurrence (attached_type);
  * every configuration attached to a detected occurrence is cyclic;
  * if every colour occurs <= 3 times: every inclusion-minimal cyclic configuration is a 3-pair
    T/B configuration (Cor. 2.10) [only for words where the full list of cyclic configs is cheap].
"""
import itertools, random, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"


def attached_config(name, wit):
    if name == "T":
        x1, x2, x3, p1, p2 = wit
        return x3, tuple(sorted([(x1, x2), p1, p2]))
    if name == "B":
        x1, x2, x3, p1, p2 = wit
        return x1, tuple(sorted([(x2, x3), p1, p2]))
    y1, y2, j, z2, z1, p = wit
    return j, tuple(sorted([(y1, z1), (y2, z2), p]))


def minimal_configs(T, allcyc):
    """inclusion-minimal cyclic configurations (same mover) among a full list."""
    by = {}
    for j, F in allcyc:
        by.setdefault(j, []).append(frozenset(F))
    out = []
    for j, L in by.items():
        for F in L:
            if not any(G < F for G in L):
                out.append((j, tuple(sorted(F))))
    return out


def run_family(label, typ, n, lengths, words=None, check_three=True, check_min=True):
    E = edges(typ, n)
    adj = make_adj(E)
    cols = nodes(typ, n)
    t0 = time.time()
    st = dict(words=0, cyclic=0, T=0, B=0, N=0, disagree=0, conn_disagree=0, three_cyclic=0,
              three_unattached=0, attached_not_cyclic=0, mult3_words=0, mult3_min_not_TB=0)
    gen = words if words is not None else (w for L in lengths for w in itertools.product(cols, repeat=L))
    for w in gen:
        w = list(w)
        st["words"] += 1
        T = Tables(w, adj)
        pats = patterns(w, E)
        p = len(pats) > 0
        b = T.brute() is not None
        c, _ = connected_search(w, adj)
        c = c is not None
        if b != c:
            st["conn_disagree"] += 1
            print("  CONN DISAGREE", label, w, b, c, flush=True)
        if b != p:
            st["disagree"] += 1
            print("  PATTERN DISAGREE", label, w, "cyclic=", b, "pattern=", p, flush=True)
        if b:
            st["cyclic"] += 1
        for nm in {nm for nm, _ in pats}:
            st[nm] += 1
        for nm, wit in pats:
            j, F = attached_config(nm, wit)
            if not is_cyclic_config(w, adj, j, F):
                st["attached_not_cyclic"] += 1
                print("  ATTACHED NOT CYCLIC", label, w, nm, wit, flush=True)
        if check_three and b:
            P = len(T.pairs)
            for j in range(len(w)):
                av = [i for i in range(P) if T.mrel[j][i] != -1]
                for F in itertools.combinations(av, 3):
                    if T.prel[F[0]][F[1]] == -1 or T.prel[F[0]][F[2]] == -1 or T.prel[F[1]][F[2]] == -1:
                        continue
                    if T.is_cyclic(j, F):
                        st["three_cyclic"] += 1
                        if not attached_type(w, E, j, [T.pairs[i] for i in F]):
                            st["three_unattached"] += 1
                            print("  3-PAIR CYCLIC NOT ATTACHED", label, w, j, [T.pairs[i] for i in F], flush=True)
        mult = max(w.count(x) for x in set(w))
        if check_min and b and mult <= 3:
            st["mult3_words"] += 1
            allc = T.brute(first_only=False)
            for j, F in minimal_configs(T, allc):
                at = attached_type(w, E, j, F) if len(F) == 3 else set()
                if len(F) != 3 or not (at & {"T", "B"}):
                    st["mult3_min_not_TB"] += 1
                    print("  MULT<=3 MINIMAL NOT T/B", label, w, j, F, at, flush=True)
    print("%-30s %s  (%.1fs)" % (label, st, time.time() - t0), flush=True)
    return st


if __name__ == "__main__":
    random.seed(20261126)
    E = edges("A", 2); adj = make_adj(E)
    w = [1, 2, 2, 1, 1, 2, 2]
    print("1221122 brute:", Tables(w, adj).brute(), " patterns:", patterns(w, E)[:3])
    w = [1, 2, 1, 2, 1, 2, 1]
    print("1212121 brute:", Tables(w, adj).brute(), " ignoring blocks:", Tables(w, adj).brute(ignore_blocks=True),
          " analyse(3,{(0,4),(1,5),(2,6)}):", analyse(w, adj, 3, [(0, 4), (1, 5), (2, 6)]), flush=True)

    Lmax = 6 if quick else 8
    run_family("A2 all, len<=%d" % Lmax, "A", 2, range(1, Lmax + 1))
    run_family("A3 all, len<=%d" % Lmax, "A", 3, range(1, Lmax + 1))
    L2 = 6 if quick else 7
    run_family("A4 all, len<=%d" % L2, "A", 4, range(1, L2 + 1))
    run_family("D4 all, len<=%d" % L2, "D", 4, range(1, L2 + 1))
    for (typ, n, L, N) in [("A", 2, 11, 150), ("A", 2, 12, 100), ("A", 3, 11, 150), ("A", 3, 12, 100),
                           ("A", 4, 11, 100), ("A", 4, 12, 80), ("D", 4, 11, 100), ("D", 4, 12, 80),
                           ("A", 5, 12, 60), ("E", 6, 12, 60)]:
        if quick:
            N = max(5, N // 10)
        cols = nodes(typ, n)
        ws = [[random.choice(cols) for _ in range(L)] for _ in range(N)]
        run_family("%s%d random len %d x%d" % (typ, n, L, N), typ, n, None, words=ws)
    for (typ, n, L, N) in [("A", 3, 12, 100), ("D", 4, 12, 100), ("A", 4, 12, 80), ("E", 6, 12, 60)]:
        if quick:
            N = max(5, N // 10)
        cols = nodes(typ, n)
        ws = []
        for _ in range(N):
            wts = [random.random() ** 2 for _ in cols]
            ws.append(random.choices(cols, weights=wts, k=L))
        run_family("%s%d biased len %d x%d" % (typ, n, L, N), typ, n, None, words=ws)
