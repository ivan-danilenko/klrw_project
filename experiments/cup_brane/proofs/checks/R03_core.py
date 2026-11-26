"""
Referee's independent implementation for proofs/03_cyclic_combinatorics.md.
Plain python3.  Does NOT import 03_lib.py or any other code of the author.

Definitions (PROOFS.md Cor. 3.1 / Prop. 7, formula.py, cyclefree.py):
  word = list of colours col[0..m-1] (slot s = index s, 0-based here)
  pair = (a, b), a < b, col[a] == col[b]
  configuration = mover j and a set F of pairwise disjoint pairs none of which contains j
  mover-p : a < j < b  ->  link if col[j] ~ c(p), block if col[j] == c(p)
  p-q     : interlaced (a_p<a_q<b_p<b_q or a_q<a_p<b_q<b_p) -> link if c(p)~c(q), block if c(p)==c(q)
  cyclic  : no block, link graph on {M} u F connected, #links > |F|
"""
import itertools
import random

# ---------------------------------------------------------------- Dynkin diagrams


def edges(typ, n):
    if typ == "A":
        E = [(i, i + 1) for i in range(1, n)]
    elif typ == "D":
        E = [(i, i + 1) for i in range(1, n - 1)] + [(n - 2, n)]
    elif typ == "E":
        E = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)][: n - 2] + [(2, 4)]
    else:
        raise ValueError(typ)
    return set(frozenset(e) for e in E)


def nodes(typ, n):
    return list(range(1, n + 1))


def make_adj(E):
    return lambda c, d: frozenset((c, d)) in E


def neighbours(E, c):
    out = []
    for e in E:
        if c in e:
            (d,) = tuple(e - {c})
            out.append(d)
    return sorted(out)

# ---------------------------------------------------------------- configurations


def interlaced(p, q):
    (a, b), (a2, b2) = p, q
    return (a < a2 < b < b2) or (a2 < a < b2 < b)


def analyse(col, adj, j, F, ignore_blocks=False):
    """Return (blocked, nlinks, connected) of the configuration (j, F), straight from the definition."""
    F = list(F)
    nv = len(F) + 1                    # vertex 0 = mover, vertex i+1 = F[i]
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    blocked = False
    nl = 0
    cj = col[j]
    for i, (a, b) in enumerate(F):
        assert a < b and col[a] == col[b] and j not in (a, b)
        if a < j < b:
            if col[a] == cj:
                blocked = True
            elif adj(col[a], cj):
                nl += 1
                union(0, i + 1)
    for i1, i2 in itertools.combinations(range(len(F)), 2):
        p, q = F[i1], F[i2]
        assert len({p[0], p[1], q[0], q[1]}) == 4
        if interlaced(p, q):
            if col[p[0]] == col[q[0]]:
                blocked = True
            elif adj(col[p[0]], col[q[0]]):
                nl += 1
                union(i1 + 1, i2 + 1)
    connected = len({find(x) for x in range(nv)}) == 1
    return blocked, nl, connected


def is_cyclic_config(col, adj, j, F, ignore_blocks=False):
    blocked, nl, conn = analyse(col, adj, j, F)
    if blocked and not ignore_blocks:
        return False
    return conn and nl > len(F)


def partial_matchings(slots):
    """All sets of pairwise disjoint pairs (a<b) on the given sorted list of slots."""
    slots = list(slots)
    if not slots:
        yield ()
        return
    first, rest = slots[0], slots[1:]
    # first unmatched
    for M in partial_matchings(rest):
        yield M
    # first matched with some later slot
    for i, s in enumerate(rest):
        others = rest[:i] + rest[i + 1:]
        for M in partial_matchings(others):
            yield ((first, s),) + M


def all_families(col, avoid):
    """All families (sets of pairwise disjoint equal-coloured pairs) avoiding the slot `avoid`."""
    classes = {}
    for s, c in enumerate(col):
        if s != avoid:
            classes.setdefault(c, []).append(s)
    lists = [list(partial_matchings(v)) for v in classes.values()]
    for combo in itertools.product(*lists):
        yield tuple(p for M in combo for p in M)


def brute_cyclic(col, adj, ignore_blocks=False, first_only=True):
    """Brute force over ALL configurations (every mover, every family). No bound on |F|."""
    found = []
    for j in range(len(col)):
        for F in all_families(col, j):
            if len(F) < 3:
                # fewer than 3 pairs: a cyclic configuration needs a cycle of length >= 3 in
                # the link graph, i.e. >= 3 vertices; with <= 2 pairs (3 vertices) a cycle would be
                # a triangle M-p-q-M.  We do NOT use this shortcut: test it anyway.
                pass
            if is_cyclic_config(col, adj, j, F, ignore_blocks):
                if first_only:
                    return (j, F)
                found.append((j, F))
    return found if not first_only else None


def connected_search(col, adj, ignore_blocks=False, stop_at_first=True, limit=None):
    """
    Exhaustive search over all connected unblocked configurations, grown one pair at a time
    (every connected configuration has an ordering of its pairs in which every prefix is
    connected -- BFS order from the mover; every sub-configuration of an unblocked one is
    unblocked).  No bound on the number of pairs.  Independent of Lemmas 2.3-2.8.
    Returns a cyclic configuration or None; also the number of configurations visited.
    """
    m = len(col)
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m) if col[a] == col[b]]
    visited = 0
    for j in range(m):
        avail = [p for p in pairs if j not in p]
        seen = {frozenset()}
        frontier = [frozenset()]
        while frontier:
            new = []
            for F in frontier:
                used = {x for p in F for x in p}
                for p in avail:
                    if p in F or p[0] in used or p[1] in used:
                        continue
                    G = F | {p}
                    if G in seen:
                        continue
                    seen.add(G)
                    blocked, nl, conn = analyse(col, adj, j, sorted(G))
                    if (blocked and not ignore_blocks) or not conn:
                        continue
                    visited += 1
                    if nl > len(G):
                        if stop_at_first:
                            return (j, tuple(sorted(G))), visited
                    new.append(G)
                    if limit is not None and visited > limit:
                        raise RuntimeError("limit")
            frontier = new
    return None, visited

# ---------------------------------------------------------------- patterns T, B, N  (text of 1.5)


def patterns(col, E):
    """
    Literal detector written from section 1.5 of the draft.  Returns a list of tuples
    (name, witness).  c' != c'' distinct neighbours of c; e ~ d ~ c (possibly e = c).
    """
    adj = make_adj(E)
    m = len(col)
    colours = sorted(set(col))
    occ = {c: [s for s in range(m) if col[s] == c] for c in colours}
    allnodes = sorted({x for e in E for x in e} | set(colours))
    nb = {c: neighbours(E, c) for c in allnodes}
    out = []

    def pairs_of(c):
        L = occ.get(c, [])
        return [(a, b) for a, b in itertools.combinations(L, 2)]

    for c in colours:
        for x1, x2, x3 in itertools.combinations(occ[c], 3):
            for c1, c2 in itertools.combinations(nb[c], 2):
                # T
                def T_ok(cc):
                    return [(y, z) for (y, z) in pairs_of(cc) if x1 < y < x2 and x3 < z]

                def B_ok(cc):
                    return [(y, z) for (y, z) in pairs_of(cc) if y < x1 and x2 < z < x3]
                t1, t2 = T_ok(c1), T_ok(c2)
                if t1 and t2:
                    out.append(("T", (x1, x2, x3, t1[0], t2[0])))
                b1, b2 = B_ok(c1), B_ok(c2)
                if b1 and b2:
                    out.append(("B", (x1, x2, x3, b1[0], b2[0])))
    # N
    for d in colours:
        for c in nb[d]:
            for e in nb[d]:              # e ~ d, possibly e == c
                for y1, y2, z2, z1 in itertools.combinations(occ[d], 4):
                    for j in occ.get(c, []):
                        if not (y2 < j < z2):
                            continue
                        for (a, b) in pairs_of(e):
                            if j in (a, b):
                                continue
                            ina, inb = (y2 < a < z2), (y2 < b < z2)
                            outa, outb = (a < y1 or a > z1), (b < y1 or b > z1)
                            if not ((ina and outb) or (inb and outa)):
                                continue
                            if e == c and a < j < b:
                                continue
                            out.append(("N", (y1, y2, j, z2, z1, (a, b))))
    return out


def has_pattern(col, E):
    return len(patterns(col, E)) > 0

# ---------------------------------------------------------------- weights, chains, heaps


def cartan(E, n):
    A = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        A[i][i] = 2
    for e in E:
        i, j = tuple(e)
        A[i][j] = A[j][i] = -1
    return A


def is_chain_word(word, E, n, k):
    """<mu_{t-1}, alpha_{c(t)}^vee> = 1 for all t, mu_0 = omega_k (fundamental-weight coords)."""
    A = cartan(E, n)
    mu = [0] * (n + 1)
    mu[k] = 1
    for c in word:
        if mu[c] != 1:
            return False, None
        for i in range(1, n + 1):
            mu[i] -= A[c][i]            # mu - alpha_c in fundamental weight coordinates
    return True, tuple(mu[1:])


def heap_key(word, E):
    """Canonical representative of the commutation class: lexicographically smallest
    linear extension of the heap (greedy).  Two letters commute iff colours distinct and non-adjacent."""
    adj = make_adj(E)
    m = len(word)
    preds = [set() for _ in range(m)]
    for t in range(m):
        for s in range(t):
            if word[s] == word[t] or adj(word[s], word[t]):
                preds[t].add(s)
    done = set()
    out = []
    while len(out) < m:
        avail = [t for t in range(m) if t not in done and preds[t] <= done]
        t = min(avail, key=lambda t: (word[t], t))
        out.append(word[t])
        done.add(t)
    return tuple(out)


def all_weights_with_heaps(E, n, k):
    """
    BFS over the orbit by chain steps.  For every weight mu, the SET of heap keys of all chain
    words omega_k -> mu (computed by DP: heaps of chains to mu = heaps of (chain to mu') + letter).
    Returns dict mu -> set of heap keys, and dict mu -> one word, and the reachability (cover) graph.
    """
    A = cartan(E, n)
    lam = [0] * (n + 1)
    lam[k] = 1
    lam = tuple(lam[1:])
    heaps = {lam: {()}}
    covers = {lam: set()}
    level = [lam]
    order = [lam]
    while level:
        new = {}
        for mu in level:
            for c in range(1, n + 1):
                if mu[c - 1] == 1:
                    nu = tuple(mu[i - 1] - A[c][i] for i in range(1, n + 1))
                    covers[mu].add(nu)
                    s = new.setdefault(nu, set())
                    for h in heaps[mu]:
                        s.add(heap_key(h + (c,), E))
        for nu, s in new.items():
            heaps[nu] = s
            covers.setdefault(nu, set())
            order.append(nu)
        level = list(new)
    return heaps, covers, order


def reachable(covers, start):
    seen = {start}
    st = [start]
    while st:
        x = st.pop()
        for y in covers[x]:
            if y not in seen:
                seen.add(y)
                st.append(y)
    return seen

# ---------------------------------------------------------------- faster table-driven brute force


class Tables:
    """Precomputed relations; same definitions as analyse()."""

    def __init__(self, col, adj):
        self.col = list(col)
        m = self.m = len(col)
        self.pairs = [(a, b) for a in range(m) for b in range(a + 1, m) if col[a] == col[b]]
        P = len(self.pairs)
        self.pc = [col[a] for a, b in self.pairs]
        # mover relation: mrel[j][i] = 0 none, 1 link, 2 block, -1 = pair contains j
        self.mrel = []
        for j in range(m):
            row = []
            for (a, b) in self.pairs:
                if j in (a, b):
                    row.append(-1)
                elif a < j < b:
                    row.append(2 if col[a] == col[j] else (1 if adj(col[a], col[j]) else 0))
                else:
                    row.append(0)
            self.mrel.append(row)
        # pair relation: prel[i][k] = 0 none, 1 link, 2 block, -1 = not disjoint
        self.prel = [[0] * P for _ in range(P)]
        for i in range(P):
            for k in range(P):
                if i == k:
                    self.prel[i][k] = -1
                    continue
                p, q = self.pairs[i], self.pairs[k]
                if len({p[0], p[1], q[0], q[1]}) < 4:
                    self.prel[i][k] = -1
                elif interlaced(p, q):
                    self.prel[i][k] = 2 if self.pc[i] == self.pc[k] else (1 if adj(self.pc[i], self.pc[k]) else 0)
        # per colour: list of slots
        self.classes = {}
        for s, c in enumerate(col):
            self.classes.setdefault(c, []).append(s)
        self.index = {p: i for i, p in enumerate(self.pairs)}

    def families(self, j):
        lists = []
        for c, sl in self.classes.items():
            sl = [s for s in sl if s != j]
            lists.append([tuple(self.index[p] for p in M) for M in partial_matchings(sl)])
        for combo in itertools.product(*lists):
            yield tuple(i for M in combo for i in M)

    def check(self, j, F, ignore_blocks=False):
        """(blocked, nlinks, connected) for the family F given as pair indices."""
        mr = self.mrel[j]
        n = len(F)
        adjl = [[] for _ in range(n + 1)]
        blocked = False
        nl = 0
        for x in range(n):
            r = mr[F[x]]
            if r == 2:
                blocked = True
                if not ignore_blocks:
                    return True, 0, False
            elif r == 1:
                nl += 1
                adjl[0].append(x + 1)
                adjl[x + 1].append(0)
        for x in range(n):
            px = self.prel[F[x]]
            for y in range(x + 1, n):
                r = px[F[y]]
                if r == 2:
                    blocked = True
                    if not ignore_blocks:
                        return True, 0, False
                elif r == 1:
                    nl += 1
                    adjl[x + 1].append(y + 1)
                    adjl[y + 1].append(x + 1)
        seen = {0}
        st = [0]
        while st:
            v = st.pop()
            for u in adjl[v]:
                if u not in seen:
                    seen.add(u)
                    st.append(u)
        return blocked, nl, len(seen) == n + 1

    def is_cyclic(self, j, F, ignore_blocks=False):
        blocked, nl, conn = self.check(j, F, ignore_blocks)
        if blocked and not ignore_blocks:
            return False
        return conn and nl > len(F)

    def brute(self, ignore_blocks=False, first_only=True):
        found = []
        for j in range(self.m):
            for F in self.families(j):
                if self.is_cyclic(j, F, ignore_blocks):
                    if first_only:
                        return (j, tuple(self.pairs[i] for i in F))
                    found.append((j, tuple(self.pairs[i] for i in F)))
        return None if first_only else found


def attached_type(col, E, j, F):
    """Is the 3-pair configuration (j, F) attached to an occurrence of T, B or N (section 1.5)?
    Returns the set of pattern names it is attached to."""
    adj = make_adj(E)
    c = col[j]
    out = set()
    F = list(F)
    for P in F:
        others = [q for q in F if q != P]
        cols_o = [col[q[0]] for q in others]
        two_nb = (cols_o[0] != cols_o[1] and adj(cols_o[0], c) and adj(cols_o[1], c))
        if col[P[0]] == c and two_nb:
            x1, x2 = P
            if x2 < j:   # T with x3 = j
                if all(x1 < y < x2 and j < z for (y, z) in others):
                    out.add("T")
            if j < x1:   # B with x1 = j, (x2, x3) = P
                if all(y < j and x1 < z < x2 for (y, z) in others):
                    out.add("B")
    for q, q2 in itertools.permutations(F, 2):
        d = col[q[0]]
        if col[q2[0]] != d or not adj(d, c):
            continue
        y1, z1 = q
        y2, z2 = q2
        if not (y1 < y2 < j < z2 < z1):
            continue
        (p,) = [r for r in F if r not in (q, q2)]
        e = col[p[0]]
        if not adj(e, d):
            continue
        a, b = p
        if j in (a, b):
            continue
        ina, inb = (y2 < a < z2), (y2 < b < z2)
        outa, outb = (a < y1 or a > z1), (b < y1 or b > z1)
        if not ((ina and outb) or (inb and outa)):
            continue
        if e == c and a < j < b:
            continue
        out.add("N")
    return out


def conn_search_fast(T, ignore_blocks=False, stop_at_first=True, collect=False):
    """Same as connected_search, table-driven.  Exhaustive over connected (unblocked unless
    ignore_blocks) configurations, grown one pair at a time, no size bound."""
    P = len(T.pairs)
    visited = 0
    found = []
    for j in range(T.m):
        avail = [i for i in range(P) if T.mrel[j][i] != -1]
        seen = {frozenset()}
        frontier = [()]
        while frontier:
            new = []
            for F in frontier:
                for i in avail:
                    if i in F:
                        continue
                    if any(T.prel[i][k] == -1 for k in F):
                        continue
                    G = frozenset(F) | {i}
                    if G in seen:
                        continue
                    seen.add(G)
                    Gt = tuple(sorted(G))
                    blocked, nl, conn = T.check(j, Gt, ignore_blocks)
                    if (blocked and not ignore_blocks) or not conn:
                        continue
                    visited += 1
                    if nl > len(Gt):
                        if stop_at_first:
                            return (j, tuple(T.pairs[k] for k in Gt)), visited
                        found.append((j, tuple(T.pairs[k] for k in Gt)))
                        if not collect:
                            continue
                    new.append(Gt)
            frontier = new
    if stop_at_first:
        return None, visited
    return found, visited
