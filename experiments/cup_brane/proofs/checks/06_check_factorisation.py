"""
For words with every colour at most twice: the closed formula for G (L.G_formula_row) equals
the composite Gamma^(1) * Gamma^(2) * ... * Gamma^(m) of the one-step maps (L.step_formula_row;
first Gamma^(1)).

    sage -python proofs/checks/06_check_factorisation.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, importlib
L = importlib.import_module("06_lib")
from collections import defaultdict


def composite(S):
    G = {I: {I: S.eI(I)} for I in L.subsets(S.m)}
    for k in range(1, S.m + 1):
        new = {}
        for I, row in G.items():
            acc = defaultdict(lambda: S.KLRW.zero())
            for J, g in row.items():
                for K, h in L.step_formula_row(S, J, k).items():
                    acc[K] += g * h
            new[I] = {K: v for K, v in acc.items() if not v.is_zero()}
        G = new
    return G


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    for red, word in zip(args[0::2], args[1::2]):
        S = L.Setup(ct, int(red), tuple(int(c) for c in word))
        G = composite(S)
        ok = True
        for I in L.subsets(S.m):
            F = L.G_formula_row(S, I)
            keys = set(F) | set(G[I])
            ok &= all((F.get(J, S.KLRW.zero()) - G[I].get(J, S.KLRW.zero())).is_zero() for J in keys)
        print("{} red {} word {}: composite of one-step maps = closed formula for G: {}".format("".join(map(str, ct)), red, word, ok), flush=True)
