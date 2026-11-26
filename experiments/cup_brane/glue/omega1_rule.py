"""
Candidate closed description of the glued complex (I-brane) for sl_{n+1}, omega_1.

Vertices: (k; A, C), 0 <= k <= n, A subset of {1..k}, C subset of {k+1..n}; B = the rest.
Idempotent: [A decreasing] W_1 [B increasing] W_n [C decreasing].

Terms out of (k; A, C):
  d = 0   one colour c <= k moves B -> A, or one colour c > k moves C -> B;
  d >= 1  target envelope l = k - d, "switching" colours l+1..k:
     (i)   all switching colours in A: they all go to C, except k which goes to B;
     (ii)  l+1 in B, the other switching colours in A: all go to C;
     (iii) l+1 in B, the others in A (for d = 1: k in B): l+1 -> C, k -> B, the others -> C
           (d = 1: k stays in B), and one further colour moves one region to the right:
           some c <= l from A to B or some c > k from B to C  [variant: any such c / only adjacent].

    python3 glue/omega1_rule.py          # compare the set of (source, target) pairs with the data
"""
import itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega1_table import load, fmt

def vertices(n):
    for k in range(n + 1):
        for a in range(k + 1):
            for A in itertools.combinations(range(1, k + 1), a):
                for c in range(n - k + 1):
                    for C in itertools.combinations(range(k + 1, n + 1), c):
                        yield (k, frozenset(A), frozenset(C))

def targets(n, v, adjacent_only=False):
    k, A, C = v
    B = frozenset(range(1, n + 1)) - A - C
    out = []
    for c in B:
        if c <= k:
            out.append(((k, A | {c}, C), ("int", c)))
    for c in C:
        out.append(((k, A, C - {c}), ("int", c)))
    for d in range(1, k + 1):
        l = k - d
        sw = list(range(l + 1, k + 1))
        if all(c in A for c in sw):                                    # (i)
            out.append(((l, A - set(sw), C | set(sw[:-1])), ("i", d)))
        if sw[0] in B and all(c in A for c in sw[1:]):                 # (ii)
            out.append(((l, A - set(sw), C | set(sw)), ("ii", d)))
            # (iii)
            if d == 1:
                A2, C2 = A, C
            else:
                A2, C2 = A - set(sw), C | set(sw[:-1])
            extras = [c for c in A if c <= l] + [c for c in B if c > k]
            for c in extras:
                if adjacent_only and not (c == l or c == k + 1):
                    continue
                if c <= l:
                    out.append(((l, A2 - {c}, C2), ("iii", d, c)))
                else:
                    out.append(((l, A2, C2 | {c}), ("iii", d, c)))
    return out

if __name__ == "__main__":
    for n in (1, 2, 3, 4, 5):
        d, P, V, terms = load(n)
        actual = set((t[0], t[1]) for t in terms)
        for adj in (False, True):
            pred = set()
            for v in vertices(n):
                for w, tag in targets(n, v, adj):
                    assert (v, w) not in pred
                    pred.add((v, w))
            print("n=%d adjacent_only=%-5s actual %4d predicted %4d missing %3d extra %3d" % (
                n, adj, len(actual), len(pred), len(actual - pred), len(pred - actual)))
            if adj and (actual - pred):
                for v, w in sorted(actual - pred, key=str)[:4]:
                    print("    missing e.g.", fmt(v, n), "->", fmt(w, n))
