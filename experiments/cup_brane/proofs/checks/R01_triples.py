"""
R01_triples.py  (referee of 01_foundations.md; pure Python, own code; plain python3)

    python3 proofs/checks/R01_triples.py [table|mult3|lambda3|all]

  table   : Lemma B.2 (crossing table), Lemma B.3 (8 triple types with middle strand), Cor. B.4
            (bad triples <=> B1-B4, independent of the sides of unaffected slots), on random
            configurations (arbitrary F, not only admissible), red colour random.
  mult3   : Theorem B.6(a),(b): exhaustive over all words with colour multiplicities <= 3
            (A_3 length <= 9, D_4 length <= 8, A_4 length <= 9 sampled), all admissible (tree,
            no block) configurations: bad triples only of type B3 in pattern S; pattern S
            gives the explicit cyclic configuration of the proof; multiplicities <= 2: none.
  lambda3 : Theorem B.6(c) for Lambda^3 C^n: every Young diagram with <= 3 rows inside the
            3 x (n-3) box (n <= 10), one linear extension each: no admissible configuration has
            a bad triple; also the first part of B.6(c) (multiplicities <= 2 for Lambda^1,2,
            so_2n vector) by direct count.
"""
import itertools
import random
import sys


def edges_of(typ, n):
    if typ == "A":
        return {frozenset((i, i + 1)) for i in range(1, n)}
    if typ == "D":
        return {frozenset((i, i + 1)) for i in range(1, n - 1)} | {frozenset((n - 2, n))}
    if typ == "E":
        E = {frozenset((1, 3)), frozenset((3, 4)), frozenset((4, 5)), frozenset((2, 4))}
        for i in range(5, n):
            E.add(frozenset((i, i + 1)))
        return E
    raise ValueError


class Word:
    def __init__(self, col, E, k=None):
        self.col = list(col)
        self.m = len(col)
        self.E = E
        self.k = k

    def c(self, s):
        return self.col[s - 1]

    def adj(self, a, b):
        return frozenset((a, b)) in self.E

    # -- configurations -------------------------------------------------------------
    def links_blocks(self, j, F):
        """returns (links, blocked) for the configuration; links: list of (u, v)."""
        links = []
        blocked = False
        cj = self.c(j)
        for p in F:
            a, b = p
            if a < j < b:
                if self.c(a) == cj:
                    blocked = True
                elif self.adj(self.c(a), cj):
                    links.append(("M", p))
        for p, q in itertools.combinations(F, 2):
            (a, b), (a2, b2) = sorted((p, q))
            if a < a2 < b < b2:
                if self.c(a) == self.c(a2):
                    blocked = True
                elif self.adj(self.c(a), self.c(a2)):
                    links.append((p, q))
        return links, blocked

    def is_tree_config(self, j, F):
        links, blocked = self.links_blocks(j, F)
        if blocked or len(links) != len(F):
            return False
        return self.connected(F, links)

    def connected(self, F, links):
        nbr = {"M": set()}
        for p in F:
            nbr[p] = set()
        for u, v in links:
            nbr[u].add(v)
            nbr[v].add(u)
        seen = {"M"}
        st = ["M"]
        while st:
            x = st.pop()
            for y in nbr[x]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        return len(seen) == len(F) + 1

    def is_cyclic_config(self, j, F):
        links, blocked = self.links_blocks(j, F)
        return (not blocked) and len(links) > len(F) and self.connected(F, links)

    def pairs(self):
        out = []
        for a in range(1, self.m + 1):
            for b in range(a + 1, self.m + 1):
                if self.c(a) == self.c(b):
                    out.append((a, b))
        return out

    def tree_configs(self, j):
        """all admissible (tree, no block) F for mover j, grown leaf by leaf (own code)."""
        avail = [p for p in self.pairs() if j not in p]
        found = {()}
        frontier = [()]
        while frontier:
            nf = []
            for F in frontier:
                used = {x for p in F for x in p}
                for p in avail:
                    if p in F or p[0] in used or p[1] in used:
                        continue
                    G = tuple(sorted(F + (p,)))
                    if G in found:
                        continue
                    if self.is_tree_config(j, G):
                        found.add(G)
                        nf.append(G)
            frontier = nf
        return found

    def is_cyclic_word(self, maxpairs=3):
        """brute force over configurations with <= maxpairs pairs (03 Thm 2.8: 3 suffice)."""
        P = self.pairs()
        for j in range(1, self.m + 1):
            av = [p for p in P if j not in p]
            for r in range(1, maxpairs + 1):
                for F in itertools.combinations(av, r):
                    sl = [x for p in F for x in p]
                    if len(set(sl)) < len(sl):
                        continue
                    if self.is_cyclic_config(j, F):
                        return (j, F)
        return None

    # -- strands and triples ---------------------------------------------------------
    def strands(self, j, F, I):
        """list of (name, colour, src, tgt)."""
        out = [("W", ("W",), 0, 0), ("M", self.c(j), j, -j)]
        used = {j}
        for p in F:
            a, b = p
            out.append((("L", p), self.c(a), -a, -b))
            out.append((("R", p), self.c(a), b, a))
            used |= {a, b}
        for y in range(1, self.m + 1):
            if y not in used:
                pos = y if y in I else -y
                out.append((("U", y), self.c(y), pos, pos))
        return out

    def triples(self, j, F, I):
        S = self.strands(j, F, I)
        cross = {}
        for A, B in itertools.combinations(range(len(S)), 2):
            sa, sb = S[A], S[B]
            cross[(A, B)] = cross[(B, A)] = (sa[2] - sb[2]) * (sa[3] - sb[3]) < 0
        out = []
        for T in itertools.combinations(range(len(S)), 3):
            if all(cross[(x, y)] for x, y in itertools.combinations(T, 2)):
                order = sorted(T, key=lambda x: S[x][2])
                out.append(tuple(S[x] for x in order))
        return S, cross, out

    def bad(self, tri):
        X, Y, Z = tri
        cx, cy, cz = X[1], Y[1], Z[1]
        if cx != cz or isinstance(cx, tuple):
            return False
        if isinstance(cy, tuple):
            return self.k is not None and cx == self.k and cy == ("W",)
        return self.adj(cx, cy)


def kind(name):
    if name in ("W", "M"):
        return name
    return name[0]


def classify(tri):
    """(type, expected middle) by the strand kinds of 01 Lemma B.3."""
    ks = tuple(sorted(kind(t[0]) for t in tri))
    tab = {("L", "M", "U"): "T1", ("M", "R", "U"): "T2", ("L", "L", "M"): "T3", ("M", "R", "R"): "T4",
           ("L", "L", "U"): "T5", ("R", "R", "U"): "T6", ("L", "L", "L"): "T7", ("R", "R", "R"): "T8"}
    return tab.get(ks)


def interval_bad(w, j, F):
    """the interval forms B1-B4 of Cor. B.4."""
    used = {j} | {x for p in F for x in p}
    U = [y for y in range(1, w.m + 1) if y not in used]
    cj = w.c(j)
    found = set()
    for p in F:
        a, b = p
        for y in U:
            if w.c(y) == cj and w.adj(cj, w.c(a)) and a < y < b < j:
                found.add("B1")
    for p in F:
        for q in F:
            if p == q:
                continue
            ap, bp = p
            aq, bq = q
            nested = ap < aq < bq < bp           # q inside p
            if not nested:
                continue
            if w.c(aq) == cj and w.adj(cj, w.c(ap)) and bp < j:
                found.add("B2")
            for y in U:
                if w.c(y) == w.c(ap) and w.adj(w.c(ap), w.c(aq)) and aq < y < bq:
                    found.add("B3")
            for r in F:
                if r in (p, q):
                    continue
                ar, br = r
                if aq < ar < br < bq and w.c(ar) == w.c(ap) and w.adj(w.c(ap), w.c(aq)):
                    found.add("B4")
    return found


def test_table(rng, N=4000):
    bad_cases = 0
    for it in range(N):
        typ, n = rng.choice([("A", 2), ("A", 3), ("A", 4), ("D", 4), ("D", 5), ("E", 6)])
        E = edges_of(typ, n)
        m = rng.randint(3, 11)
        col = [rng.randint(1, n) for _ in range(m)]
        w = Word(col, E, k=rng.randint(1, n))
        j = rng.randint(1, m)
        # arbitrary disjoint same-colour pairs avoiding j
        P = [p for p in w.pairs() if j not in p]
        rng.shuffle(P)
        F = []
        used = {j}
        for p in P:
            if p[0] not in used and p[1] not in used and rng.random() < 0.5:
                F.append(p)
                used |= set(p)
        F = tuple(sorted(F))
        I = {j} | {b for a, b in F}
        for y in range(1, m + 1):
            if y not in used and rng.random() < 0.5:
                I.add(y)
        S, cross, tris = w.triples(j, F, I)
        names = [s[0] for s in S]
        # Lemma B.2
        def cr(x, y):
            return cross[(names.index(x), names.index(y))]
        for A, B in itertools.combinations(names, 2):
            ka, kb = kind(A), kind(B)
            got = cr(A, B)
            exp = None
            pair = {ka, kb}
            if pair == {"M", "W"}:
                exp = True
            elif "W" in pair:
                exp = False
            elif pair == {"M", "U"}:
                y = (A if ka == "U" else B)[1]
                exp = y < j
            elif pair == {"M", "L"} or pair == {"M", "R"}:
                p = (A if ka != "M" else B)[1]
                exp = p[1] < j
            elif pair == {"L", "U"} or pair == {"R", "U"}:
                X = A if ka != "U" else B
                y = (A if ka == "U" else B)[1]
                p = X[1]
                side_ok = (y not in I) if kind(X) == "L" else (y in I)
                exp = side_ok and p[0] < y < p[1]
            elif pair == {"U"}:
                exp = False
            elif pair == {"L", "R"}:
                exp = False
            elif ka == kb and ka in ("L", "R"):
                p, q = A[1], B[1]
                exp = (p[0] < q[0] < q[1] < p[1]) or (q[0] < p[0] < p[1] < q[1])
            assert exp is not None, (A, B)
            assert got == exp, ("B.2", col, j, F, A, B)
        # Lemma B.3
        for tri in tris:
            typ3 = classify(tri)
            assert typ3 is not None, ("B.3 unexpected triple", col, j, F, [t[0] for t in tri])
            mid = tri[1][0]
            if typ3 in ("T1", "T2"):
                assert kind(mid) in ("L", "R"), tri
            elif typ3 in ("T3", "T4"):
                outer = [t for t in (tri[0], tri[2]) if t[0] != "M"][0]
                # middle is the outer (containing) pair
                assert kind(mid) in ("L", "R") and mid[1][0] < outer[0][1][0] < outer[0][1][1] < mid[1][1], tri
            elif typ3 in ("T5", "T6"):
                LL = [t for t in tri if kind(t[0]) != "U"]
                inner = min(LL, key=lambda t: t[0][1][1] - t[0][1][0])
                assert mid == inner[0], tri
            else:
                ivs = sorted((t[0][1] for t in tri), key=lambda p: p[1] - p[0])
                assert mid[1] == ivs[1], tri
            assert all(t[0] != "W" for t in tri)
        # Cor. B.4
        bad_brute = any(w.bad(t) for t in tris)
        types = set()
        for t in tris:
            if w.bad(t):
                types.add({"T1": "B1", "T2": "B1", "T3": "B2", "T4": "B2", "T5": "B3", "T6": "B3",
                           "T7": "B4", "T8": "B4"}[classify(t)])
        iv = interval_bad(w, j, F)
        assert types == iv, ("B.4", col, j, F, types, iv)
        assert bad_brute == bool(iv)
        bad_cases += bad_brute
    print("table: %d random configurations: Lemma B.2, Lemma B.3 (types, middle strands, W in no "
          "triple), Cor. B.4 (types per configuration) all confirmed; %d had bad triples" % (N, bad_cases))
    return True


def words_mult(n_colours, length, maxmult):
    for col in itertools.product(range(1, n_colours + 1), repeat=length):
        if max(col.count(c) for c in set(col)) <= maxmult:
            yield col


def patternS(w, j, F, parent):
    """detect pattern S literally (01 Thm B.6(b)) and return the cyclic configuration of the proof."""
    cj = w.c(j)
    for p in F:
        for q in F:
            if p == q:
                continue
            ap, bp = p
            aq, bq = q
            if not (ap < aq < bq < bp):
                continue
            if w.c(aq) != cj or not w.adj(w.c(ap), cj):
                continue
            # tree path q - x - M - p
            if parent.get(p) != "M":
                continue
            x = parent.get(q)
            if x in (None, "M") or parent.get(x) != "M":
                continue
            d = w.c(x[0])
            if not w.adj(d, cj) or d == w.c(ap):
                continue
            used = {j} | {s for r in F for s in r}
            for y in range(aq + 1, bq):
                if y in used or w.c(y) != w.c(ap):
                    continue
                pstar = (ap, y) if j < aq else (y, bp)
                return (j, tuple(sorted((q, x, pstar))))
    return None


def tree_parent(w, j, F):
    links, _ = w.links_blocks(j, F)
    nbr = {"M": []}
    for p in F:
        nbr[p] = []
    for u, v in links:
        nbr[u].append(v)
        nbr[v].append(u)
    parent = {}
    seen = {"M"}
    st = ["M"]
    while st:
        x = st.pop()
        for y in nbr[x]:
            if y not in seen:
                seen.add(y)
                parent[y] = x
                st.append(y)
    return parent


def test_mult3(rng):
    ok = True
    runs = [("A", 3, 9, 3, None), ("D", 4, 8, 3, None), ("A", 4, 9, 3, 60000), ("A", 3, 8, 2, None),
            ("D", 4, 8, 2, None)]
    for typ, n, L, mm, sample in runs:
        E = edges_of(typ, n)
        words = [wd for length in range(3, L + 1) for wd in words_mult(n, length, mm)]
        if sample is not None and len(words) > sample:
            words = rng.sample(words, sample)
        n_bad = n_badwords = n_S = 0
        other = 0
        for col in words:
            w = Word(col, E, k=1)
            word_bad = False
            for j in range(1, w.m + 1):
                for F in w.tree_configs(j):
                    if not F:
                        continue
                    I = {j} | {b for a, b in F}
                    S, cross, tris = w.triples(j, F, I)
                    bads = [t for t in tris if w.bad(t)]
                    if not bads:
                        continue
                    word_bad = True
                    n_bad += 1
                    types = {classify(t) for t in bads}
                    if mm <= 2 or not types <= {"T5", "T6"}:
                        other += 1
                        ok = False
                        print("  UNEXPECTED", typ, n, col, j, F, types)
                        continue
                    cyc = patternS(w, j, F, tree_parent(w, j, F))
                    if cyc is None:
                        ok = False
                        print("  B3 bad triple NOT in pattern S", col, j, F)
                        continue
                    n_S += 1
                    jj, FF = cyc
                    if not w.is_cyclic_config(jj, FF):
                        ok = False
                        print("  pattern S config not cyclic", col, cyc)
            n_badwords += word_bad
        print("mult3: %s_%d, words of length <= %d with multiplicities <= %d: %d words; %d words / %d "
              "admissible configurations with a bad triple; all of type B3 in pattern S with the "
              "proof's cyclic configuration verified: %d; unexpected: %d"
              % (typ, n, L, mm, len(words), n_badwords, n_bad, n_S, other))
    return ok


def young_diagrams(rows, cols):
    """partitions with <= rows parts, each <= cols."""
    def rec(r, maxpart):
        if r == 0:
            yield ()
            return
        for a in range(maxpart, -1, -1):
            for rest in rec(r - 1, a):
                yield (a,) + rest
    for lam in rec(rows, cols):
        yield tuple(x for x in lam if x > 0)


def word_of_diagram(lam, k, rng):
    """random linear extension of the boxes (r,s) (componentwise order), colour k + s - r."""
    boxes = [(r, s) for r in range(1, len(lam) + 1) for s in range(1, lam[r - 1] + 1)]
    remaining = set(boxes)
    order = []
    while remaining:
        minimal = [b for b in remaining if not any(c != b and c[0] <= b[0] and c[1] <= b[1] for c in remaining)]
        b = rng.choice(sorted(minimal))
        order.append(b)
        remaining.remove(b)
    return [k + s - r for (r, s) in order]


def test_lambda3(rng):
    ok = True
    for n in range(6, 11):
        k = 3
        E = edges_of("A", n - 1)
        cnt = nb = 0
        for lam in young_diagrams(3, n - k):
            if not lam:
                continue
            col = word_of_diagram(lam, k, rng)
            w = Word(col, E, k=k)
            cnt += 1
            for j in range(1, w.m + 1):
                for F in w.tree_configs(j):
                    if not F:
                        continue
                    I = {j} | {b for a, b in F}
                    _, _, tris = w.triples(j, F, I)
                    if any(w.bad(t) for t in tris):
                        nb += 1
                        ok = False
                        print("  BAD in Lambda^3", n, lam, col, j, F)
        print("lambda3: sl_%d, Lambda^3: %d Young diagrams (one random linear extension each), "
              "admissible configurations with a bad triple: %d" % (n, cnt, nb))
    # multiplicities for Lambda^2 and so_2n vector
    mx2 = max(max(col.count(c) for c in set(col)) for n in range(4, 11)
              for lam in young_diagrams(2, n - 2) if lam
              for col in [word_of_diagram(lam, 2, rng)])
    print("lambda3: max colour multiplicity over all Lambda^2 C^n ideals, n <= 10: %d" % mx2)
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    rng = random.Random(5)
    ok = True
    if mode in ("table", "all"):
        ok &= test_table(rng)
    if mode in ("lambda3", "all"):
        ok &= test_lambda3(rng)
    if mode in ("mult3", "all"):
        ok &= test_mult3(rng)
    print("ALL OK" if ok else "FAILURES")
