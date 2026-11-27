"""
Finer splitting of the group adj(j) of 07_conjecture_H.md section 5.1 by configuration (F, a):
F = the family of pairs of the prefix term with the pair through m removed, a = partner of m.
Reports which configurations vanish on their own; the failing ones (nested families) are printed.

    sage -python 07_adjfine.py      (runs the words used in section 5.5)
"""
import sys, importlib, itertools
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("07_lab")
G = importlib.import_module("07_groups"); G.ADJ_BY_J = True
from collections import defaultdict
def key_of(S, label, prods):
    m = S.m
    if label == "zF":
        T = prods[1]; j, F = T.info
        (p,) = [q for q in F if q[1] == m]
        return (j, tuple(sorted(q for q in F if q != p)), p[0], "A5")
    if label == "dR.h":
        T, H = prods; j, F = T.info; c = H.info[0][0]
        for p in F:
            if p[1] == c: return (j, tuple(sorted(q for q in F if q != p)), p[0], "A3b")
        return (j, F, c, "A3" if F else "A1")
    H, T = prods; j, F = T.info; c = H.info[0][0]
    for p in F:
        if p[1] == c: return (j, tuple(sorted(q for q in F if q != p)), p[0], "A4b")
    return (j, F, c, "A4" if F else "A2")
def run(ct, red, word):
    S = L.Lab(ct, red, word); m = S.m
    st = defaultdict(lambda: [0, 0]); bad = []
    for h in range(1, m+1):
        for rest in itertools.combinations(range(1, m), h-1):
            I = frozenset(rest) | {m}
            for (K, key), lst in G.groups(S, I).items():
                if not (isinstance(key, tuple) and key[0] == "J"): continue
                sub = defaultdict(list)
                for label, prods, sgn in lst:
                    k = key_of(S, label, prods)
                    sub[k[:3]].append((label, prods, sgn, k[3]))
                for k, l2 in sub.items():
                    tot = defaultdict(lambda: S.F.zero())
                    for label, prods, sgn, _ in l2:
                        for t, f in S.expand(I, prods).items(): tot[t] += sgn*f
                    ok = all(v == 0 for v in tot.values())
                    j, F, a = k
                    cls = ("a<j" if a < j else "a>j", "F" if F else "noF")
                    st[cls][0 if ok else 1] += 1
                    if not ok: bad.append((I, K, k, sorted(set(x[3] for x in l2))))
    print("".join(map(str, word)), dict(st)); 
    for b in bad[:6]: print("   ", b)
run(["A",2],2,(1,2,1,1)); run(["A",2],2,(1,1,2,1,1,1)); run(["A",3],2,(1,1,2,1,3,1,1)); run(["D",4],2,(2,1,3,2,4,2,2)); run(["A",2],1,(1,1,1,2,1,1,1))
