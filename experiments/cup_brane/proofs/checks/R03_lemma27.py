"""
Referee test of Lemma 2.5 (replacement) and Lemma 2.7 (cycle shortening) on random unblocked
families, independently of the author's code.

  python3 R03_lemma27.py [N]

Random words (types A_3..A_6, D_4, D_5, E_6; length 18-30), random unblocked families grown by
adding random equal-coloured disjoint pairs that create no block.  In the link graph G(F) of the
family (pairs only) all chordless cycles of length >= 6 (up to length 12) are enumerated; at every
turning point the surgery of Lemma 2.7 is applied literally and all its conclusions are checked:
  x* is a pair of colour d; F' = (C - {x,y,z}) u {x*} is a family (disjoint) and unblocked;
  G(F') contains the cycle (x*, v_{i+2}, ..., v_{i-2}).
Also tested: Lemma 2.5 on every chordless cycle (any length >= 4) found.
"""
import random, sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import edges, make_adj, interlaced, nodes


def rel(col, adj, p, q):
    if not interlaced(p, q):
        return 0
    if col[p[0]] == col[q[0]]:
        return 2
    return 1 if adj(col[p[0]], col[q[0]]) else 0


def chi(p, s):
    return 1 if p[0] < s < p[1] else 0


def random_family(col, adj, tries=400):
    m = len(col)
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m) if col[a] == col[b]]
    random.shuffle(pairs)
    F = []
    used = set()
    for p in pairs[:tries]:
        if p[0] in used or p[1] in used:
            continue
        if any(rel(col, adj, p, q) == 2 for q in F):
            continue
        F.append(p)
        used |= set(p)
    return F


def chordless_cycles(F, A, Lmin, Lmax):
    """All chordless cycles (as vertex lists, canonical up to rotation/reflection) of the graph
    with adjacency sets A on vertices 0..len(F)-1, lengths in [Lmin, Lmax]."""
    n = len(F)
    out = set()
    for s in range(n):
        # cycles whose minimal vertex is s
        stack = [(s, [s])]
        while stack:
            v, path = stack.pop()
            for u in A[v]:
                if u < s:
                    continue
                if u == s and len(path) >= Lmin:
                    # check chordless
                    cyc = path
                    L = len(cyc)
                    ok = True
                    for i in range(L):
                        for k in range(i + 2, L):
                            if i == 0 and k == L - 1:
                                continue
                            if cyc[k] in A[cyc[i]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        r = cyc[1:]
                        key = tuple(cyc) if cyc[1] < cyc[-1] else tuple([cyc[0]] + r[::-1])
                        out.add(key)
                    continue
                if u in path or len(path) >= Lmax:
                    continue
                # prune: new vertex must not be adjacent to interior path vertices (chordless)
                if any(u in A[w] for w in path[1:-1]):
                    continue
                if len(path) >= 2 and u in A[path[0]] and False:
                    pass
                stack.append((u, path + [u]))
    return [list(c) for c in out]


GRID = "grid" in sys.argv


def main(N):
    random.seed(7)
    types = [("A", 3), ("A", 4), ("A", 5), ("A", 6), ("D", 4), ("D", 5), ("E", 6)]
    stats = dict(families=0, cycles_ge6=0, surgeries=0, failures=0, len={}, lemma25=0, lemma25_fail=0)
    for it in range(N):
        if GRID and it % 2 == 0:
            # word of a B x B box (type A_{2B-1}, colour B + s - r), row-major
            B = random.choice([5, 6, 7])
            E = edges("A", 2 * B - 1); adj = make_adj(E)
            col = [B + s - r for r in range(1, B + 1) for s in range(1, B + 1)]
            if random.random() < 0.5:
                keep = sorted(random.sample(range(len(col)), random.randint(20, len(col))))
                col = [col[i] for i in keep]
        else:
            typ, n = random.choice(types)
            E = edges(typ, n); adj = make_adj(E)
            L = random.randint(18, 30)
            cols = nodes(typ, n)
            wts = [random.random() + 0.2 for _ in cols]
            col = random.choices(cols, weights=wts, k=L)
        F = random_family(col, adj)
        stats["families"] += 1
        A = [set() for _ in F]
        for i, k in itertools.combinations(range(len(F)), 2):
            if rel(col, adj, F[i], F[k]) == 1:
                A[i].add(k); A[k].add(i)
        cycs = chordless_cycles(F, A, 4, 12)
        for cyc in cycs:
            C = [F[i] for i in cyc]
            Lc = len(C)
            # Lemma 2.5
            stats["lemma25"] += 1
            p = max(C, key=lambda q: q[1])
            j = p[0]
            rest = [q for q in C if q != p]
            ok = True
            for r in rest:
                # mover relation with r must equal relation of p with r
                if r[0] < j < r[1]:
                    mr = 2 if col[r[0]] == col[j] else (1 if adj(col[r[0]], col[j]) else 0)
                else:
                    mr = 0
                if mr != rel(col, adj, p, r):
                    ok = False
            if not ok:
                stats["lemma25_fail"] += 1
                print("LEMMA 2.5 FAIL", col, C)
            if Lc < 6:
                continue
            stats["cycles_ge6"] += 1
            stats["len"][Lc] = stats["len"].get(Lc, 0) + 1
            for i in range(Lc):
                x, z, y = C[(i - 1) % Lc], C[i], C[(i + 1) % Lc]
                d = col[x[0]]
                if col[y[0]] != d:
                    continue                      # not a turning point
                u2, w2 = C[(i - 2) % Lc], C[(i + 2) % Lc]
                kappa = chi(u2, y[0])
                kappa2 = chi(w2, x[0])
                ss = [s for s in x if chi(u2, s) == 1 - kappa]
                tt = [t for t in y if chi(w2, t) == 1 - kappa2]
                stats["surgeries"] += 1
                bad = []
                if len(ss) != 1 or len(tt) != 1:
                    bad.append("s,t not unique")
                else:
                    s, t = ss[0], tt[0]
                    xs = (min(s, t), max(s, t))
                    if col[xs[0]] != d or col[xs[1]] != d or s == t:
                        bad.append("x* colour")
                    Fp = [q for q in C if q not in (x, y, z)] + [xs]
                    sl = [a for q in Fp for a in q]
                    if len(set(sl)) != len(sl):
                        bad.append("not disjoint")
                    for q1, q2 in itertools.combinations(Fp, 2):
                        if rel(col, adj, q1, q2) == 2:
                            bad.append("block %s %s" % (q1, q2))
                    newcyc = [xs] + [C[(i + 2 + k) % Lc] for k in range(Lc - 3)]
                    for k in range(len(newcyc)):
                        if rel(col, adj, newcyc[k], newcyc[(k + 1) % len(newcyc)]) != 1:
                            bad.append("missing link %d" % k)
                if bad:
                    stats["failures"] += 1
                    print("LEMMA 2.7 FAIL", col, C, i, bad)
    print(stats)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
    # usage: python3 R03_lemma27.py N [grid]
