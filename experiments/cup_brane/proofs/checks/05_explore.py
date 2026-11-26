"""
Tabulate every product of two tree terms occurring in d^2, expanded in the basis.

    sage -python 05_explore.py TYPE RANK WORD [v]

Prints, for every pair of vertices (I, K) with a non-trivial contribution, the products and the
cancellation pattern (grouped by resulting basis element).
"""
import sys, importlib
from collections import defaultdict
L = importlib.import_module("05_lib")


def explore(ct, word, verbose=True, red=1, out=sys.stdout):
    S = L.Setup(ct, red, word)
    m = S.m
    stats = defaultdict(int)
    import itertools
    for h in range(2, m + 1):
        for I in itertools.combinations(range(1, m + 1), h):
            I = frozenset(I)
            byK = defaultdict(list)
            for t1 in S.terms(I):
                x = S.el(I, t1)
                for t2 in S.terms(t1.J):
                    byK[t2.J].append((t1, t2, x * S.el(t1.J, t2)))
            for K, prods in sorted(byK.items(), key=lambda kv: sorted(kv[0])):
                groups = defaultdict(list)
                total = S.K.zero()
                for t1, t2, p in prods:
                    total += p
                    d = S.describe(I, K, p)
                    stats["products"] += 1
                    stats["products with %d basis terms" % len(d)] += 1
                    for key, co in d.items():
                        groups[key].append((t1, t2, co))
                ok = total.is_zero()
                stats["(I,K) ok" if ok else "(I,K) FAIL"] += 1
                sizes = tuple(sorted(len(g) for g in groups.values()))
                stats["pattern %s" % (sizes,)] += 1
                if verbose and (len(prods) > 2 or not ok):
                    print("I=%s K=%s  %s" % (sorted(I), sorted(K), "ok" if ok else "FAIL"), file=out)
                    for key, g in groups.items():
                        print("    [%s]" % L.fmt_key(key), file=out)
                        for t1, t2, co in g:
                            print("        %-22s * %-22s : %s" % (L.fmt_term(t1), L.fmt_term(t2), co), file=out)
    return stats


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    word = tuple(int(c) for c in sys.argv[3])
    stats = explore(ct, word, verbose=len(sys.argv) > 4)
    for k, v in sorted(stats.items()):
        print("%-40s %d" % (k, v))
