"""
Output comparison with the author's 03_lib / 03_patterns (imported ONLY here, to compare answers).
    python3 R03_compare_author.py
"""
import sys, os, random, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *
L = importlib.import_module("03_lib")
P = importlib.import_module("03_patterns")

random.seed(5)
bad = 0; n_cyc = 0; N = 0
for t in range(3000):
    typ, n = random.choice([("A", 2), ("A", 3), ("A", 4), ("D", 4), ("E", 6), ("D", 5)])
    E = edges(typ, n); adj = make_adj(E)
    w = [random.choice(nodes(typ, n)) for _ in range(random.randint(5, 13))]
    mine = conn_search_fast(Tables(w, adj))[0] is not None
    mine_p = has_pattern(w, E)
    W = L.Word(w, L.adjacency(typ, n))
    theirs = W.search() is not None if not isinstance(W.search(), bool) else W.search()
    theirs_p = bool(P.patterns(w, L.adjacency(typ, n)))
    N += 1; n_cyc += mine
    if len({mine, mine_p, bool(theirs), theirs_p}) > 1:
        bad += 1
        print("DIFF", typ, n, w, mine, mine_p, theirs, theirs_p)
print("compared %d random words (%d cyclic): disagreements %d" % (N, n_cyc, bad))
