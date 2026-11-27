"""
(star) of 06 Prop. 5.2 with the Conjecture H formula, expanded in the labelled basis by 07_lab.
Reports, for every vertex I containing m, whether all coefficients vanish.

    sage -python 07_star_check.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, importlib, time
L = importlib.import_module("07_lab")


def check(ct, red, word, verbose=False):
    t0 = time.time()
    S = L.Lab(ct, red, word)
    bad = []
    n = 0
    for h in range(1, S.m + 1):
        import itertools
        for rest in itertools.combinations(range(1, S.m), h - 1):
            I = frozenset(rest) | {S.m}
            tab = S.star_table(I)
            for K, row in tab.items():
                for tau, lst in row.items():
                    n += 1
                    tot = sum(c for _, _, c in lst)
                    if tot != 0:
                        bad.append((I, K, tau, tot))
    print("{} red {} word {}: {} (I,K,tau) classes, {} nonzero  [{:.1f}s]".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), n, len(bad), time.time() - t0), flush=True)
    if verbose:
        for I, K, tau, tot in bad[:10]:
            print("   ", L.fmt(I), "->", L.fmt(K), L.cycles(tau), tot.factor())
    return not bad


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    v = "v" in args
    args = [a for a in args if a != "v"]
    for red, word in zip(args[0::2], args[1::2]):
        check(ct, int(red), tuple(int(c) for c in word), v)
