"""
Shared pure-Python library for  proofs/03_cyclic_combinatorics.md  (the cyclic condition).

No sage, no klrw.  Import from another script in this folder with

    import importlib, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    L = importlib.import_module("03_lib")

Conventions (PROOFS.md Cor. 3.1, Prop. 7).  A word is a list of colours ``col[0..m-1]``
(slot s = index s, 0-based here).  ``adj(c, d)`` says whether two colours are joined in the
Dynkin diagram.  A configuration is a mover ``j`` and a set ``F`` of pairwise disjoint pairs
``(a < b)`` of equal-coloured slots, none containing ``j``.

    link  mover-p :  a_p < j < b_p  and  col[j] ~ c(p)
    block mover-p :  a_p < j < b_p  and  col[j] = c(p)
    link  p-q     :  interlaced      and  c(p) ~ c(q)
    block p-q     :  interlaced      and  c(p) = c(q)

A configuration is *cyclic* if it has no block, its link graph on {mover} u F is connected and
#links > #pairs.  A word is cyclic if it admits a cyclic configuration.

Search method (exhaustive, no bound on the number of pairs).  Lemma 2 of the write-up:
every cyclic configuration contains a cyclic configuration of the form  T + {p}  where T is a
*tree configuration* (no block, link graph a tree) and p is a pair, disjoint from T and
from the mover, not blocked by T or by the mover, with at least two links into T.  Every tree
configuration is obtained from the bare mover by adding leaves one at a time.  So
``search`` enumerates all tree configurations of every mover (breadth first, with a seen-set,
no size bound) and tests every candidate pair against each of them.
"""
import itertools
from collections import defaultdict

# ----------------------------------------------------------------------------- Dynkin data

def dynkin_edges(typ, n):
    """Bourbaki labelling.  'Z' = the infinite path (colours are integers, c ~ c+1)."""
    if typ == "A":
        E = [(i, i + 1) for i in range(1, n)]
    elif typ == "D":
        E = [(i, i + 1) for i in range(1, n - 1)] + [(n - 2, n)]
    elif typ == "E":
        E = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)][: n - 2] + [(2, 4)]
    else:
        raise ValueError(typ)
    return set(frozenset(e) for e in E)


def adjacency(typ, n=None):
    if typ == "Z":
        return lambda c, d: abs(c - d) == 1
    E = dynkin_edges(typ, n)
    return lambda c, d: frozenset((c, d)) in E


def cartan(typ, n):
    E = dynkin_edges(typ, n)
    A = [[0] * (n + 1) for _ in range(n + 1)]          # 1-based
    for i in range(1, n + 1):
        A[i][i] = 2
    for e in E:
        i, j = tuple(e)
        A[i][j] = A[j][i] = -1
    return A

# ----------------------------------------------------------------------------- minuscule data

def minuscule_full_word(typ, n, k):
    """A chain omega_k -> lowest weight: each step subtracts alpha_i with <mu, alpha_i^vee> = 1.
    Weights in fundamental-weight coordinates (1-based tuple, index 0 unused).  Greedy, smallest i."""
    A = cartan(typ, n)
    mu = [0] * (n + 1)
    mu[k] = 1
    word = []
    while True:
        down = [i for i in range(1, n + 1) if mu[i] == 1]
        if not down:
            break
        i = down[0]
        word.append(i)
        c = mu[i]
        for j in range(1, n + 1):
            mu[j] -= c * A[i][j]            # simply laced: alpha_i = sum_j A_ij omega_j
    return word


def weight_orbit(typ, n, k):
    """All weights of V(omega_k) reachable by chains (= the W-orbit for minuscule omega_k)."""
    A = cartan(typ, n)
    lam = tuple(1 if j == k else 0 for j in range(n + 1))
    seen = {lam}
    frontier = [lam]
    while frontier:
        new = []
        for mu in frontier:
            for i in range(1, n + 1):
                if mu[i] == 1:
                    nu = tuple(mu[j] - A[i][j] for j in range(n + 1))
                    if nu not in seen:
                        seen.add(nu)
                        new.append(nu)
        frontier = new
    return seen


def chain_weights(typ, n, k, word):
    """Weights along the chain of ``word`` from omega_k; raises if a step is not a chain step."""
    A = cartan(typ, n)
    mu = [1 if j == k else 0 for j in range(n + 1)]
    out = [tuple(mu)]
    for i in word:
        if mu[i] != 1:
            raise ValueError("not a minuscule chain")
        for j in range(1, n + 1):
            mu[j] -= A[i][j]
        out.append(tuple(mu))
    return out


def heap_order(col, adj):
    """Heap of a word: x < y iff x precedes y and there is a chain of slots with equal or
    adjacent consecutive colours.  Returns ``less[x]`` = set of y with x < y."""
    m = len(col)
    cover = [[y for y in range(x + 1, m) if col[x] == col[y] or adj(col[x], col[y])] for x in range(m)]
    less = [set() for _ in range(m)]
    for x in range(m - 1, -1, -1):
        for y in cover[x]:
            less[x].add(y)
            less[x] |= less[y]
    return less


def order_ideals(m, less):
    """All order ideals (down-sets) of the heap on slots 0..m-1, as sorted tuples."""
    below = [set() for _ in range(m)]
    for x in range(m):
        for y in less[x]:
            below[y].add(x)
    ideals = []

    def rec(i, cur):
        if i == m:
            ideals.append(tuple(sorted(cur)))
            return
        rec(i + 1, cur)                       # slot i not in the ideal
        if below[i] <= cur:                   # every slot below i is in (slots below i are < i)
            cur.add(i)
            rec(i + 1, cur)
            cur.discard(i)

    rec(0, set())
    return ideals

# ----------------------------------------------------------------------------- configurations

NONE, LINK, BLOCK, CLASH = 0, 1, 2, 3


class Word:
    def __init__(self, col, adj):
        self.col = list(col)
        self.m = len(self.col)
        self.adj = adj
        bycol = defaultdict(list)
        for s, c in enumerate(self.col):
            bycol[c].append(s)
        self.bycol = bycol
        self.pairs = [(a, b) for sl in bycol.values() for a, b in itertools.combinations(sl, 2)]
        self.pairs.sort()
        P = self.pairs
        self.np = len(P)
        # pair-pair relations as bitmasks
        self.linkmask = [0] * self.np
        self.blockmask = [0] * self.np
        self.clashmask = [0] * self.np
        for i, (a, b) in enumerate(P):
            for k, (a2, b2) in enumerate(P):
                if k == i:
                    continue
                r = self.pair_rel((a, b), (a2, b2))
                if r == LINK:
                    self.linkmask[i] |= 1 << k
                elif r == BLOCK:
                    self.blockmask[i] |= 1 << k
                elif r == CLASH:
                    self.clashmask[i] |= 1 << k

    def same(self, s, t):
        return self.col[s] == self.col[t]

    def adjc(self, s, t):
        return self.adj(self.col[s], self.col[t])

    def pair_rel(self, p, q):
        (a, b), (a2, b2) = p, q
        if len({a, b, a2, b2}) < 4:
            return CLASH
        if (a < a2 < b < b2) or (a2 < a < b2 < b):
            if self.col[a] == self.col[a2]:
                return BLOCK
            if self.adj(self.col[a], self.col[a2]):
                return LINK
        return NONE

    def mover_rel(self, j, p):
        a, b = p
        if j in (a, b):
            return CLASH
        if a < j < b:
            if self.col[a] == self.col[j]:
                return BLOCK
            if self.adj(self.col[a], self.col[j]):
                return LINK
        return NONE

    # -- direct evaluation of a configuration (used by the brute force and for checking) ----
    def graph(self, j, F):
        """(edges, blocks) of the configuration (j, F); vertices 'm' and the pairs."""
        edges, blocks = [], []
        for p in F:
            r = self.mover_rel(j, p)
            assert r != CLASH
            if r == LINK:
                edges.append(("m", p))
            elif r == BLOCK:
                blocks.append(("m", p))
        for p, q in itertools.combinations(F, 2):
            r = self.pair_rel(p, q)
            assert r != CLASH
            if r == LINK:
                edges.append((p, q))
            elif r == BLOCK:
                blocks.append((p, q))
        return edges, blocks

    def is_cyclic_config(self, j, F, allow_blocks=False):
        edges, blocks = self.graph(j, F)
        if blocks and not allow_blocks:
            return False
        return len(edges) > len(F) and connected(["m"] + list(F), edges)

    # -- exhaustive search ----------------------------------------------------------------
    def search(self, allow_blocks=False, mover_on_cycle=False, collect=False, movers=None):
        """
        Exhaustive search for cyclic configurations.
        Returns the first witness (j, F) found, or None; with ``collect=True`` returns the
        list of ALL configurations T+{p} (T a tree configuration, p with >= 2 links into T)
        -- this list contains every inclusion-minimal cyclic configuration (Lemma 2).
        ``mover_on_cycle``: only accept p linked to the mover (then the configuration has a
        cycle through the mover; conversely every configuration with a cycle through the
        mover contains such a T+{p}, Lemma 3).
        ``allow_blocks``: ignore blocks altogether (the "blocked" variant).
        Also returns the number of tree configurations visited (``self.ntrees``).
        """
        if mover_on_cycle and not collect:
            return self._search_paths(allow_blocks, movers)
        P = self.pairs
        found = []
        self.ntrees = 0
        for j in (range(self.m) if movers is None else movers):
            mrel = [self.mover_rel(j, p) for p in P]
            seen = {0}
            frontier = [0]
            while frontier:
                new = []
                for T in frontier:
                    self.ntrees += 1
                    used = 0
                    for i in bits(T):
                        used |= 1 << P[i][0] | 1 << P[i][1]
                    for i, (a, b) in enumerate(P):
                        if T >> i & 1 or mrel[i] == CLASH or (used >> a & 1) or (used >> b & 1):
                            continue
                        if not allow_blocks and (mrel[i] == BLOCK or self.blockmask[i] & T):
                            continue
                        nl = (mrel[i] == LINK) + popcount(self.linkmask[i] & T)
                        if nl == 0:
                            continue
                        U = T | 1 << i
                        if nl == 1:
                            if U not in seen:
                                seen.add(U)
                                new.append(U)
                            continue
                        if mover_on_cycle and mrel[i] != LINK:
                            continue
                        F = tuple(P[x] for x in bits(U))
                        if not collect:
                            return (j, F)
                        found.append((j, F))
                frontier = new
        return found if collect else None

    def _search_paths(self, allow_blocks=False, movers=None):
        """Cycle through the mover.  Lemma 3: such a configuration exists iff there are an
        induced path  mover - v_1 - ... - v_r  (no blocks; v_{i+1} linked to v_i only) and a
        pair p, unblocked against it, linked to the mover and to at least one v_i.
        Enumerates induced paths only (state = (vertex set, end)), no length bound."""
        P = self.pairs
        self.ntrees = 0
        for j in (range(self.m) if movers is None else movers):
            mrel = [self.mover_rel(j, p) for p in P]
            seen = {(0, -1)}
            frontier = [(0, -1)]                  # end = -1 means: the path is the bare mover
            while frontier:
                new = []
                for T, end in frontier:
                    self.ntrees += 1
                    used = 0
                    for i in bits(T):
                        used |= 1 << P[i][0] | 1 << P[i][1]
                    for i, (a, b) in enumerate(P):
                        if T >> i & 1 or mrel[i] == CLASH or (used >> a & 1) or (used >> b & 1):
                            continue
                        if not allow_blocks and (mrel[i] == BLOCK or self.blockmask[i] & T):
                            continue
                        lm = mrel[i] == LINK
                        lt = popcount(self.linkmask[i] & T)
                        if lm and (lt >= 1):
                            return (j, tuple(P[x] for x in bits(T | 1 << i)))
                        # extend the induced path at its end
                        if end == -1:
                            ok = lm and lt == 0
                        else:
                            ok = (not lm) and lt == 1 and (self.linkmask[i] >> end & 1)
                        if ok and (T | 1 << i, i) not in seen:
                            seen.add((T | 1 << i, i))
                            new.append((T | 1 << i, i))
                frontier = new
        return None

    def minimal_cyclic(self, allow_blocks=False):
        """All inclusion-minimal cyclic configurations (set of (j, F) with F sorted)."""
        cand = set(self.search(allow_blocks=allow_blocks, collect=True))
        out = set()
        for j, F in cand:
            if all(not self.is_cyclic_config(j, tuple(q for q in F if q != p), allow_blocks)
                   for p in F):
                out.add((j, tuple(sorted(F))))
        return out

    # -- brute force over all configurations (independent check, small words only) --------
    def brute(self, allow_blocks=False, mover_on_cycle=False):
        for j in range(self.m):
            avail = [p for p in self.pairs if j not in p]
            for F in matchings(avail):
                if not F:
                    continue
                if not self.is_cyclic_config(j, F, allow_blocks):
                    continue
                if mover_on_cycle and not mover_on_some_cycle(j, F, self):
                    continue
                return (j, F)
        return None


def mover_on_some_cycle(j, F, W):
    """Does the link graph of (j, F) have a cycle through the mover?  (A vertex v lies on a
    cycle iff two of its neighbours are connected in the graph minus v.)"""
    edges, _ = W.graph(j, F)
    nb = [q for e in edges if "m" in e for q in e if q != "m"]
    rest = [e for e in edges if "m" not in e]
    for x, y in itertools.combinations(nb, 2):
        if connected_pair(x, y, list(F), rest):
            return True
    return False


def connected_pair(x, y, V, E):
    adjl = defaultdict(set)
    for u, v in E:
        adjl[u].add(v)
        adjl[v].add(u)
    seen = {x}
    st = [x]
    while st:
        u = st.pop()
        for v in adjl[u]:
            if v not in seen:
                seen.add(v)
                st.append(v)
    return y in seen


def connected(V, E):
    if not V:
        return True
    adjl = defaultdict(set)
    for u, v in E:
        adjl[u].add(v)
        adjl[v].add(u)
    seen = {V[0]}
    st = [V[0]]
    while st:
        u = st.pop()
        for v in adjl[u]:
            if v not in seen:
                seen.add(v)
                st.append(v)
    return len(seen) == len(V)


def matchings(pairs):
    """All sets of pairwise disjoint pairs from the list (including the empty set)."""
    out = []

    def rec(i, cur, used):
        if i == len(pairs):
            out.append(tuple(cur))
            return
        rec(i + 1, cur, used)
        a, b = pairs[i]
        if a not in used and b not in used:
            cur.append(pairs[i])
            used.add(a); used.add(b)
            rec(i + 1, cur, used)
            cur.pop()
            used.discard(a); used.discard(b)

    rec(0, [], set())
    return out


def bits(x):
    i = 0
    while x:
        if x & 1:
            yield i
        x >>= 1
        i += 1


def popcount(x):
    return bin(x).count("1")

# ----------------------------------------------------------------------------- shape of a configuration

def shape(W, j, F):
    """Describe a (minimal) cyclic configuration: cycle length, distance of the mover from the
    cycle, colours along the cycle.  Assumes the link graph is a chordless cycle with a
    pendant path (Lemma 2)."""
    edges, _ = W.graph(j, F)
    V = ["m"] + list(F)
    adjl = defaultdict(set)
    for u, v in edges:
        adjl[u].add(v); adjl[v].add(u)
    # 2-core: strip vertices of degree <= 1 repeatedly; what is left is the cycle part
    alive = set(V)
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            if len(adjl[v] & alive) <= 1:
                alive.discard(v)
                changed = True
    cyc = alive
    dist = {"m": 0}
    st = ["m"]
    while st:
        u = st.pop(0)
        for v in adjl[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                st.append(v)
    d = min(dist[v] for v in cyc)
    colour = lambda v: W.col[j] if v == "m" else W.col[v[0]]
    chordless = all(len(adjl[v] & cyc) == 2 for v in cyc)
    walk = []
    if chordless:
        start = "m" if "m" in cyc else min(cyc)
        walk = [start]
        prev, cur = None, start
        while True:
            nxt = sorted((v for v in adjl[cur] & cyc if v != prev), key=str)
            v = nxt[0]
            if v == start:
                break
            walk.append(v)
            prev, cur = cur, v
    return {"cycle_len": len(cyc), "mover_dist": d, "npairs": len(F),
            "cycle_colours": tuple(colour(v) for v in walk), "chordless": chordless}
