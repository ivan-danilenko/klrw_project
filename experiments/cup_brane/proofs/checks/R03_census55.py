"""Census of inclusion-minimal cyclic configurations of the 5x5 square (brute force over all configurations)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *
from R03_misc import link_graph, is_tadpole_from_M
from R03_typeA import word_rowmajor
t0 = time.time()
k = 6
E = edges("A", 11); adj = make_adj(E)
w = word_rowmajor((5, 5, 5, 5, 5), k)
T = Tables(w, adj)
allc = T.brute(first_only=False)
by = {}
for j, F in allc:
    by.setdefault(j, []).append(frozenset(F))
mins = [(j, tuple(sorted(F))) for j, L in by.items() for F in L if not any(G < F for G in L)]
shapes = {}
bad = 0
for j, F in mins:
    ok, sh = is_tadpole_from_M(link_graph(w, adj, j, F))
    if ok:
        shapes[sh] = shapes.get(sh, 0) + 1
    else:
        bad += 1
print("5^5: %d cyclic configurations, %d inclusion-minimal, not tadpole %d, (cycle,tail) %s  (%.0fs)" % (
    len(allc), len(mins), bad, sorted(shapes.items()), time.time() - t0))
