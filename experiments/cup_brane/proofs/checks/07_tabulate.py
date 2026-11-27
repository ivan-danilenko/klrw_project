"""
Tabulate every product in (star) at the vertices I (m in I), expanded in the labelled basis and
grouped by the resulting labelled permutation tau.

    sage -python 07_tabulate.py TYPE RANK RED WORD [I as comma list]
"""
import sys, importlib, itertools
L = importlib.import_module("07_lab")


def show(S, I):
    tab = S.star_table(I)
    for K in sorted(tab, key=sorted):
        for tau, lst in sorted(tab[K].items()):
            tot = sum(c for _, _, c in lst)
            print("  {} -> {}  tau {}   total {}".format(L.fmt(I), L.fmt(K), L.cycles(tau), tot))
            for label, prods, c in lst:
                print("      {:6s} {:40s} {}".format(label, " * ".join(map(repr, prods)), c.factor() if c != 0 else 0))


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3]); word = tuple(int(c) for c in sys.argv[4])
    S = L.Lab(ct, red, word)
    if len(sys.argv) > 5:
        show(S, frozenset(int(a) for a in sys.argv[5].split(",")))
    else:
        for h in range(1, S.m + 1):
            for rest in itertools.combinations(range(1, S.m), h - 1):
                show(S, frozenset(rest) | {S.m})
