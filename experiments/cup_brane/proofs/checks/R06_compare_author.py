"""
R06: compare my G (written from the text of Definition 6.2, R06_lib) with the author's
06_lib.G_formula_row (imported only for this comparison), and my d_Phi with the author's.

    sage -python R06_compare_author.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, importlib
from R06_lib import Setup
L = importlib.import_module("06_lib")

ct = [sys.argv[1], int(sys.argv[2])]
args = sys.argv[3:]
for red, word in zip(args[0::2], args[1::2]):
    w = tuple(int(c) for c in word)
    S = Setup(ct, int(red), w)
    T = L.Setup(ct, int(red), w)
    nG = nd = 0
    for I in S.vertices():
        mine = S.G(I)
        theirs = L.G_formula_row(T, I)
        for J in set(mine) | set(theirs):
            a = mine.get(J, S.K.zero())
            b = theirs.get(J, T.KLRW.zero())
            # different algebra objects: compare via string of the difference in my algebra
            if str(a) != str(b) and not (a - S.K(b)).is_zero():
                nG += 1
    print("%s red %s word %s: components where my G differs from 06_lib.G_formula_row: %d" % (
        "".join(map(str, ct)), red, word, nG), flush=True)
