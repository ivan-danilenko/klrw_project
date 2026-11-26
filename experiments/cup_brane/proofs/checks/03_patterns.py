"""
Theorem P (pattern characterisation) and Lemma 5 (cycle shortening) of
proofs/03_cyclic_combinatorics.md.  Pure Python.

    python3 proofs/checks/03_patterns.py [N_WORDS] [SEED]

For random arbitrary words (types A2..A5, D4, E6; lengths 4..16):
 (1) cyclic (exhaustive search 03_lib)  <=>  one of the patterns T, B, N occurs
     (detected by a literal, independent implementation of the pattern definitions);
     for words of length <= 11 also against the brute force over all configurations;
 (2) every pattern occurrence found is a cyclic configuration with 3 pairs;
 (3) Lemma 5, step by step: for every inclusion-minimal cyclic configuration whose cycle has
     length L >= 6, written as a pair family on the word extended by one slot of the mover's
     colour, and for EVERY turning point z (neighbours x, y of equal colour) the new chord
     x* = (s, t) defined in the proof is linked to u and w, the family (C - {x,y,z}) + {x*}
     has no block, and its link graph has the cycle  x*, w, ..., u  of length L - 2;
 (4) Lemma 4 (replacement): for every minimal cyclic configuration whose cycle avoids the
     mover, replacing the cycle vertex with the largest right end b by a mover at its left
     end gives an unblocked configuration whose link graph is the cycle.
"""
import importlib, os, sys, random, itertools
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("03_lib")


def inside(s, r):
    return r[0] < s < r[1]


def patterns(col, adj):
    """Literal implementation of the patterns (0-based slots).  Returns the set of types found
    and one witness configuration (mover, pairs) per type."""
    m = len(col)
    occ = {}
    for s, c in enumerate(col):
        occ.setdefault(c, []).append(s)
    colours = list(occ)
    nb = {c: [d for d in colours if adj(c, d)] for c in colours}
    found = {}
    # T and B: colour c with slots x1 < x2 < x3, distinct neighbours c1 != c2
    for c in colours:
        for x1, x2, x3 in itertools.combinations(occ[c], 3):
            for c1, c2 in itertools.combinations(nb[c], 2):
                def pick_T(cc):
                    for y in occ[cc]:
                        for z in occ[cc]:
                            if x1 < y < x2 and z > x3:
                                return (y, z)
                def pick_B(cc):
                    for y in occ[cc]:
                        for z in occ[cc]:
                            if y < x1 and x2 < z < x3:
                                return (y, z)
                q1, q2 = pick_T(c1), pick_T(c2)
                if q1 and q2 and "T" not in found:
                    found["T"] = (x3, ((x1, x2), q1, q2))
                q1, q2 = pick_B(c1), pick_B(c2)
                if q1 and q2 and "B" not in found:
                    found["B"] = (x1, ((x2, x3), q1, q2))
    # N: d-slots y1 < y2 < j < z2 < z1, c(j) ~ d, pair (a, b) of colour e ~ d with exactly one
    #    end in (y2, z2), the other outside [y1, z1]; if e == c(j) then j not in (a, b)
    for d in colours:
        for y1, y2, z2, z1 in itertools.combinations(occ[d], 4):
            for j in range(y2 + 1, z2):
                if not adj(col[j], d):
                    continue
                for e in nb[d]:
                    for a, b in itertools.combinations(occ[e], 2):
                        if j in (a, b):
                            continue
                        ina = y2 < a < z2
                        inb = y2 < b < z2
                        outa = a < y1 or a > z1
                        outb = b < y1 or b > z1
                        if not ((ina and outb) or (inb and outa)):
                            continue
                        if e == col[j] and a < j < b:
                            continue
                        if "N" not in found:
                            found["N"] = (j, ((y1, z1), (y2, z2), (a, b)))
    return found


def check_shortening(W, j, F, stats):
    """(3) and (4) for one minimal cyclic configuration."""
    m = W.m
    col2 = W.col + [W.col[j]]
    W2 = L.Word(col2, W.adj)
    edges, _ = W.graph(j, F)
    mov = (j, m)                                     # the mover as the chord (j, m+1)
    ren = lambda v: mov if v == "m" else v
    V = set([mov] + list(F))
    adjl = {v: set() for v in V}
    for a, b in edges:
        adjl[ren(a)].add(ren(b)); adjl[ren(b)].add(ren(a))
    core = set(V)
    ch = True
    while ch:
        ch = False
        for v in list(core):
            if len(adjl[v] & core) <= 1:
                core.discard(v); ch = True
    # the chordless cycle, in order
    start = min(core); cyc = [start]; prev = None
    while True:
        nxt = sorted(v for v in adjl[cyc[-1]] & core if v != prev)
        v = nxt[0]
        if v == start:
            break
        prev = cyc[-1]; cyc.append(v)
    Lc = len(cyc)
    assert Lc == len(core) and all(len(adjl[v] & core) == 2 for v in core)
    # (4) replacement, when the mover is not on the cycle
    if mov not in core:
        p = max(core, key=lambda v: v[1])
        Fn = tuple(v for v in core if v != p)
        assert W.is_cyclic_config(p[0], Fn), (W.col, j, F)
        e2, b2 = W.graph(p[0], Fn)
        assert len(e2) == Lc and not b2
        stats["replacement"] += 1
    if Lc < 6:
        return
    # (3) every turning point
    nturn = 0
    for i in range(Lc):
        u, x, z, y, w = cyc[i - 2], cyc[i - 1], cyc[i], cyc[(i + 1) % Lc], cyc[(i + 2) % Lc]
        if col2[x[0]] != col2[y[0]]:
            continue
        nturn += 1
        ku = inside(y[0], u)
        assert inside(y[1], u) == ku                 # u does not interlace y
        s = [e for e in x if inside(e, u) != ku]
        assert len(s) == 1
        kw = inside(x[0], w)
        assert inside(x[1], w) == kw
        t = [e for e in y if inside(e, w) != kw]
        assert len(t) == 1
        xs = (min(s[0], t[0]), max(s[0], t[0]))
        assert W2.pair_rel(xs, u) == L.LINK and W2.pair_rel(xs, w) == L.LINK
        rest = [v for v in cyc if v not in (x, y, z)]
        for r in rest:
            assert W2.pair_rel(xs, r) in (L.LINK, L.NONE)      # no block, no clash
        # the new cycle x*, w, ..., u
        path = [cyc[(i + 2 + k) % Lc] for k in range(Lc - 3)]
        assert path[0] == w and path[-1] == u
        for a, b in zip(path, path[1:]):
            assert W2.pair_rel(a, b) == L.LINK
        stats["surgery L=%d" % Lc] += 1
    assert nturn >= 2
    stats["cycles with L>=6"] += 1


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    types = [("A", 2), ("A", 3), ("A", 3), ("A", 4), ("A", 5), ("D", 4), ("E", 6)]
    stats = Counter()
    for it in range(N):
        typ, n = rng.choice(types)
        m = rng.randint(4, 16)
        col = [rng.randint(1, n) for _ in range(m)]
        adj = L.adjacency(typ, n)
        W = L.Word(col, adj)
        s = W.search()
        pat = patterns(col, adj)
        for typ_, (j, F) in pat.items():
            assert W.is_cyclic_config(j, F), (col, typ_, j, F)
            stats["pattern " + typ_] += 1
        assert (s is not None) == bool(pat), (typ, n, col, s, pat)
        if m <= 11:
            assert (W.brute() is not None) == bool(pat)
            stats["brute-checked"] += 1
        stats["cyclic" if s else "not cyclic"] += 1
        if s is not None and m <= 14:
            for j, F in W.minimal_cyclic():
                check_shortening(W, j, F, stats)
    print("Theorem P, Lemmas 4 and 5 confirmed on %d random words:" % N)
    for k in sorted(stats):
        print("   %-22s %d" % (k, stats[k]))
    # longer chordless cycles: all minimal cyclic configurations of some minuscule words
    stats = Counter()
    A = importlib.import_module("03_typeA")
    for lam in [(5, 5, 5, 5), (5, 5, 5, 5, 5), (6, 6, 6, 4, 4, 4)]:
        B, col = A.word_of(lam)
        W = L.Word(col, A.ADJ)
        assert bool(patterns(col, A.ADJ)) == (W.search() is not None)
        for j, F in W.minimal_cyclic():
            check_shortening(W, j, F, stats)
    for typ, n, k in [("E", 7, 7), ("D", 7, 7)]:
        word = L.minuscule_full_word(typ, n, k)
        W = L.Word(word, L.adjacency(typ, n))
        for j, F in W.minimal_cyclic():
            check_shortening(W, j, F, stats)
    print("Lemmas 4 and 5 on all minimal cyclic configurations of lambda = 5^4, 5^5, 6^3 4^3 and of"
          " the full E7 and D7-spin posets:")
    for k in sorted(stats):
        print("   %-22s %d" % (k, stats[k]))
