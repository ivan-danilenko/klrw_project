"""
01_mult3.py   (python3 proofs/checks/01_mult3.py TYPE RANK MAXLEN MAXMULT)

Theorem B.6 (a),(b) of 01_foundations.md: over all words with every colour occurring at most
MAXMULT times, count bad triples of admissible configurations by type (B1)-(B4).
"""
import sys, itertools, importlib.util
from collections import Counter
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("bs", os.path.join(HERE, "01_bad_search.py"))
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
from formula import Stack
ct = [sys.argv[1], int(sys.argv[2])]; L = int(sys.argv[3]); M = int(sys.argv[4])
cnt = Counter(); ex = {}
nw = 0
for LL in range(3, L + 1):
    for w in itertools.product(range(1, ct[1] + 1), repeat=LL):
        if max(Counter(w).values()) > M: continue
        nw += 1
        st = Stack(ct, w)
        for j, F in bs.configurations(st):
            for b in bs.bad_triples(st, j, F):
                cnt[b[0]] += 1
                ex.setdefault(b[0], ("".join(map(str, w)), j, F, b))
print(ct, "words up to length", L, "with colour multiplicity <=", M, ":", nw, "words; bad triples by type:", dict(cnt))
for k, v in ex.items(): print("  example", k, v)
