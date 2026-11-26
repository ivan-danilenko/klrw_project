"""
Referee checks of smaller claims: Lemma 2.3(a) tadpole shape of inclusion-minimal configurations,
the 6-cycle example of section 3.3, the T/B/N census of the 6x6 box, Lemma 2.12 / Cor. 2.13 /
Lemma 2.14 on minuscule chain words.     python3 R03_misc.py
"""
import sys, os, itertools, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R03_core import *
from R03_typeA import partitions_in_box, boxes, word_rowmajor, contains


def link_graph(col, adj, j, F):
    V = ["M"] + list(F)
    A = {v: set() for v in V}
    for p in F:
        if p[0] < j < p[1] and adj(col[p[0]], col[j]):
            A["M"].add(p); A[p].add("M")
    for p, q in itertools.combinations(F, 2):
        if interlaced(p, q) and adj(col[p[0]], col[q[0]]):
            A[p].add(q); A[q].add(p)
    return A


def is_tadpole_from_M(A):
    """cycle C (len>=4) + path M=v0..vl with V(C) cap V(P) = {vl}, no other edges.
    Equivalently: connected, |E| = |V|, and either all degrees 2 (l = 0), or the degree sequence is
    one vertex of degree 1 (= M), one of degree 3, all others 2."""
    V = list(A)
    nE = sum(len(s) for s in A.values()) // 2
    seen = {"M"}; st = ["M"]
    while st:
        x = st.pop()
        for y in A[x]:
            if y not in seen:
                seen.add(y); st.append(y)
    if len(seen) != len(V) or nE != len(V):
        return False, None
    deg = {v: len(A[v]) for v in V}
    d1 = [v for v in V if deg[v] == 1]
    d3 = [v for v in V if deg[v] == 3]
    if all(deg[v] == 2 for v in V):
        return (len(V) >= 4), (len(V), 0)
    if d1 == ["M"] and len(d3) == 1 and all(deg[v] in (1, 2, 3) for v in V):
        # tail length = distance from M to the degree-3 vertex
        dist = {"M": 0}; q = ["M"]
        while q:
            x = q.pop(0)
            for y in A[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1; q.append(y)
        l = dist[d3[0]]
        return (len(V) - l >= 4), (len(V) - l, l)
    return False, None


if __name__ == "__main__":
    random.seed(3)
    # Lemma 2.3(a) on (5,5,5,5) and (5,5,5,5,5)? (the latter is big: brute force of all configs)
    k = 6
    E = edges("A", 11); adj = make_adj(E)
    for lam in [(4, 4, 4, 4), (5, 5, 5, 5), (6, 5, 5, 5)]:
        w = word_rowmajor(lam, k)
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
            if not ok:
                bad += 1
            else:
                shapes[sh] = shapes.get(sh, 0) + 1
        # converse: unblocked configurations whose link graph is a tadpole from M are minimal
        conv_bad = 0
        minset = set(mins)
        for j, F in allc:
            ok, sh = is_tadpole_from_M(link_graph(w, adj, j, F))
            if ok and (j, tuple(sorted(F))) not in minset:
                conv_bad += 1
        print("%s: %d cyclic, %d minimal; minimal not tadpole: %d; tadpole not minimal: %d; (cycle length, tail length): %s" % (
            lam, len(allc), len(mins), bad, conv_bad, sorted(shapes.items())), flush=True)
    # the 6-cycle example in (5,5,5,5)
    lam = (5, 5, 5, 5)
    bx = boxes(lam); w = word_rowmajor(lam, k); ix = {b: i for i, b in enumerate(bx)}
    j = ix[(2, 2)]
    F = [((1, 2), (2, 3)), ((1, 3), (3, 5)), ((3, 4), (4, 5)), ((3, 3), (4, 4)), ((2, 1), (4, 3))]
    Fi = [tuple(sorted((ix[a], ix[b]))) for a, b in F]
    A = link_graph(w, adj, j, Fi)
    print("6-cycle example: analyse =", analyse(w, adj, j, Fi), " degrees:", sorted(len(s) for s in A.values()),
          " tadpole/cycle:", is_tadpole_from_M(A))
    # T/B/N census in the 6x6 box
    nT = nB = nN = ncyc = 0
    for lam in partitions_in_box(6, 6):
        lam = tuple(x for x in lam if x > 0)
        w = word_rowmajor(lam, k)
        names = {nm for nm, _ in patterns(w, E)}
        if names:
            ncyc += 1
            nT += "T" in names; nB += "B" in names; nN += "N" in names
    print("6x6 box: cyclic %d, with T %d, with B %d, with N %d" % (ncyc, nT, nB, nN))
    # Lemma 2.12, Cor 2.13, Lemma 2.14 on all minuscule chain words (one per weight)
    for typ, n, kk in [("A", 9, 5), ("A", 11, 6), ("D", 9, 9), ("D", 9, 1), ("E", 6, 1), ("E", 6, 6), ("E", 7, 7)]:
        E = edges(typ, n); adj = make_adj(E)
        heaps, covers, order = all_weights_with_heaps(E, n, kk)
        full = max((next(iter(h)) for h in heaps.values()), key=len)
        fullmult = [full.count(c) for c in range(1, n + 1)]
        bad12 = bad13 = badB = bad14 = 0
        for mu in order:
            (w,) = heaps[mu]
            w = list(w)
            for c in range(1, n + 1):
                occ = [t for t in range(len(w)) if w[t] == c]
                if not occ:
                    continue
                nbr = lambda t: adj(w[t], c)
                if sum(1 for t in range(occ[0]) if nbr(t)) != (0 if c == kk else 1):
                    bad12 += 1
                for t1, t2 in zip(occ, occ[1:]):
                    if sum(1 for t in range(t1 + 1, t2) if nbr(t)) != 2:
                        bad12 += 1
            if any(w.count(c) > fullmult[c - 1] for c in range(1, n + 1)):
                bad14 += 1
            for nm, wit in patterns(w, E):
                if nm == "B":
                    x1 = wit[0]
                    if all(w[t] != w[x1] for t in range(x1)):
                        badB += 1
            if max([w.count(c) for c in set(w)] + [0]) <= 3:
                # Cor 2.13 criterion
                crit = False
                for c in set(w):
                    occ = [t for t in range(len(w)) if w[t] == c]
                    if len(occ) != 3:
                        continue
                    x1, x2, x3 = occ
                    between = [w[t] for t in range(x1 + 1, x2) if adj(w[t], c)]
                    if len(between) == 2 and between[0] != between[1] and \
                            all(any(w[t] == cc for t in range(x3 + 1, len(w))) for cc in between):
                        crit = True
                cyc = conn_search_fast(Tables(w, adj))[0] is not None
                if crit != cyc:
                    bad13 += 1
        print("%s%d omega_%d: %d weights; Lemma 2.12 violations %d; Lemma 2.14 violations %d; B at a first occurrence %d; Cor 2.13 disagreements %d" % (
            typ, n, kk, len(order), bad12, bad14, badB, bad13), flush=True)
