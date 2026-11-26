"""
Referee checks for proofs/04_first_order.md -- pure-Python part (plain python3).

Written independently of 04_*.py, degree_model.py and 04_lib.py: own Dynkin diagrams, own
grading group (PROOFS.md F2, coordinates q, sig_c, del_c, eps_{cc'}, epsW_c), own positions and
crossing tests, own reduced words (bubble sort, NOT the klrw insertion sort).  The only import
from the project is formula.Stack, the authoritative statement of the tree formula and its signs.

    python3 proofs/checks/R04_comb.py prop53  [N]     Prop. 5.3 crossing classification
    python3 proofs/checks/R04_comb.py signs   [N]     Lemma 5.1 and the sign identities of Thm C
    python3 proofs/checks/R04_comb.py defect  [N]     Prop. 7.1 (degree defect) vs direct grading
    python3 proofs/checks/R04_comb.py prop74  [N]     Prop. 7.4 (mult <= 2), (a) and (b)
    python3 proofs/checks/R04_comb.py cyc7            Thm 9.1(a) counts, cyclic configurations of 1221122
    python3 proofs/checks/R04_comb.py gauge WORD RED TYPE RANK   homogeneous elements of hom. degree 0
"""
import sys, os, itertools, random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from formula import Stack  # noqa: E402  (authoritative signs)


# ----------------------------------------------------------------------------- Dynkin
def dynkin(ct):
    t, n = ct
    if t == "A":
        E = [(i, i + 1) for i in range(1, n)]
    elif t == "D":
        E = [(i, i + 1) for i in range(1, n - 1)] + [(n - 2, n)]
    elif t == "E":
        E = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)][: n - 2] + [(2, 4)]
    else:
        raise ValueError(t)
    return {frozenset(e) for e in E}


class W:
    """A word (slots 1..m) with red colour `red` in type `ct`."""

    def __init__(self, ct, red, word):
        self.ct = list(ct)
        self.red = red
        self.col = tuple(word)
        self.m = len(word)
        self.E = dynkin(ct)
        self.slots = tuple(range(1, self.m + 1))

    def c(self, s):
        return self.col[s - 1]

    def adj(self, a, b):
        return frozenset((a, b)) in self.E

    # ------------------------------------------------------------- positions / states
    def labels(self, I):
        I = set(I)
        return [s for s in sorted(set(self.slots) - I, reverse=True)] + ["W"] + sorted(I)

    @staticmethod
    def pos(I, s):
        if s == "W":
            return 0
        return s if s in I else -s

    def endpoints(self, I, J, sigma):
        """strand label -> (source position, target position); sigma: dict slot -> slot."""
        out = {"W": (0, 0)}
        for s in self.slots:
            out[s] = (self.pos(I, s), self.pos(J, sigma[s]))
        return out

    def inversions(self, I, J, sigma):
        ep = self.endpoints(I, J, sigma)
        keys = list(ep)
        inv = set()
        for u, v in itertools.combinations(keys, 2):
            (a1, b1), (a2, b2) = ep[u], ep[v]
            if (a1 - a2) * (b1 - b2) < 0:
                inv.add(frozenset((u, v)))
        return inv

    def bubble_word(self, I, J, sigma):
        """Own reduced word (1-based letters, first letter first): repeatedly swap the
        LEFTMOST adjacent inverted pair.  Returns (word, list of crossing pairs in order)."""
        lab = self.labels(I)
        tgt = {s: self.pos(J, sigma[s]) for s in self.slots}
        tgt["W"] = 0
        word, pairs = [], []
        while True:
            for p in range(len(lab) - 1):
                if tgt[lab[p]] > tgt[lab[p + 1]]:
                    pairs.append(frozenset((lab[p], lab[p + 1])))
                    lab[p], lab[p + 1] = lab[p + 1], lab[p]
                    word.append(p + 1)
                    break
            else:
                break
        tl = [tgt[x] for x in lab]
        assert tl == sorted(tl) and len(word) == len(self.inversions(I, J, sigma))
        return tuple(word), pairs

    # ------------------------------------------------------------- grading (F2)
    def xdeg(self, u, v):
        """degree of a crossing of strands of colours u, v ('W' = red)."""
        if u == "W" or v == "W":
            c = v if u == "W" else u
            return Counter({"q": 1, ("epsW", c): 1}) if c == self.red else Counter()
        if u == v:
            return Counter({"q": -2, ("sig", u): 1})
        if self.adj(u, v):
            return Counter({"q": 1, ("eps", min(u, v), max(u, v)): 1})
        return Counter()

    def colour(self, lab):
        return "W" if lab == "W" else self.c(lab)

    def dx(self, x):
        d = Counter(self.xdeg(self.c(x), "W"))
        for i in range(1, x):
            d.update(self.xdeg(self.c(i), self.c(x)))
        return d

    def Dv(self, I):
        d = Counter()
        for x in I:
            d.update(self.dx(x))
        return d

    def degbeta(self, I, J, sigma):
        d = Counter()
        for pr in self.inversions(I, J, sigma):
            u, v = tuple(pr)
            d.update(self.xdeg(self.colour(u), self.colour(v)))
        return d

    def defect(self, I, J, sigma):
        d = Counter(self.Dv(I))
        d.subtract(self.Dv(J))
        d.subtract(self.degbeta(I, J, sigma))
        return Counter({k: v for k, v in d.items() if v})

    def solve(self, delta):
        """All (n, m, l, a) with deg(prod r^n t^m u^l * (a_c dots of colour c)) = delta.
        Direct comparison of coefficients in the free abelian group (no use of Prop. 7.1)."""
        cols = sorted(set(self.col))
        allc = sorted({c for e in self.E for c in e} | set(cols) | {self.red})
        for k in delta:
            if k == "q":
                continue
            if k[0] in ("sig", "del", "epsW"):
                continue
            if k[0] == "eps":
                continue
            return []
        n = {c: delta.get(("sig", c), 0) for c in allc}
        if any(v < 0 for v in n.values()):
            return []
        l = {}
        for c in allc:
            v = delta.get(("epsW", c), 0)
            if v < 0 or v % 2:
                return []
            l[c] = v // 2
        edges = sorted(tuple(sorted(e)) for e in self.E)
        tot = []
        for a, b in edges:
            v = delta.get(("eps", a, b), 0)
            if v < 0 or v % 2:
                return []
            tot.append(v // 2)
        out = []
        for split in itertools.product(*[range(h + 1) for h in tot]):
            m = {}
            for (a, b), h, k in zip(edges, tot, split):
                m[a, b] = k
                m[b, a] = h - k
            a_ = {}
            ok = True
            for c in allc:
                # del_c:  n_c - sum_b m_cb - l_c + a_c = delta[del_c]
                a_[c] = delta.get(("del", c), 0) - n[c] + sum(v for (x, y), v in m.items() if x == c) + l[c]
                if a_[c] < 0 or (a_[c] > 0 and c not in cols):
                    ok = False
            if not ok:
                continue
            if 2 * sum(a_.values()) != delta.get("q", 0):
                continue
            out.append(({c: v for c, v in n.items() if v}, {k: v for k, v in m.items() if v},
                        {c: v for c, v in l.items() if v}, {c: v for c, v in a_.items() if v}))
        return out

    def bijections(self):
        classes = defaultdict(list)
        for s in self.slots:
            classes[self.c(s)].append(s)
        cls = list(classes.values())
        for perms in itertools.product(*[itertools.permutations(c) for c in cls]):
            sigma = {}
            for cl, p in zip(cls, perms):
                for a, b in zip(cl, p):
                    sigma[a] = b
            yield sigma

    def subsets(self):
        for h in range(self.m + 1):
            for c in itertools.combinations(self.slots, h):
                yield frozenset(c)


def sigma_inv(sigma):
    return {v: k for k, v in sigma.items()}


def ident(m):
    return {s: s for s in range(1, m + 1)}


def transp(m, pairs):
    s = ident(m)
    for a, b in pairs:
        s[a], s[b] = b, a
    return s


def compose(s1, s2):
    """first s1 then s2 (slot maps)"""
    return {k: s2[v] for k, v in s1.items()}


def nI(I, x):
    return sum(1 for z in I if z > x)


def NI(I, a, b):
    return sum(1 for z in I if a < z < b)


# ----------------------------------------------------------------------------- words
def random_words(rng, ct, n, minlen, maxlen, maxmult=None, force_equal_neighbour=False):
    nodes = sorted({c for e in dynkin(ct) for c in e})
    out = set()
    tries = 0
    while len(out) < n and tries < 100 * n:
        tries += 1
        L = rng.randint(minlen, maxlen)
        w = tuple(rng.choice(nodes) for _ in range(L))
        if maxmult and max(Counter(w).values()) > maxmult:
            continue
        if force_equal_neighbour and not any(w[i] == w[i + 1] for i in range(L - 1)):
            continue
        out.add(w)
    return sorted(out)


TYPES = [["A", 4], ["D", 4], ["D", 5], ["E", 6]]


# ----------------------------------------------------------------------------- Prop 5.3
def one_pair_configs(Wd, I):
    out = []
    for j in I:
        for a in Wd.slots:
            if a in I:
                continue
            for b in I:
                if b == j or not (a < j < b) or Wd.c(a) != Wd.c(b) or not Wd.adj(Wd.c(j), Wd.c(a)):
                    continue
                out.append((j, (a, b)))
    return out


def crossing_counts_concat(Wd, I, J, s1, K, s2):
    """crossing numbers of (reduced word of s1: I->J) . (reduced word of s2: J->K), strands
    labelled by their source label at I."""
    cnt = Counter()
    for pr in Wd.inversions(I, J, s1):
        cnt[pr] += 1
    # strands of the second factor are labelled by their slot in J; translate back
    back = {v: k for k, v in s1.items()}
    back["W"] = "W"
    for pr in Wd.inversions(J, K, s2):
        u, v = tuple(pr)
        cnt[frozenset((back[u], back[v]))] += 1
    return cnt


def check_prop53(nw, seed=1):
    rng = random.Random(seed)
    stats = Counter()
    fails = 0
    for ct in TYPES + [["A", 2], ["A", 3]]:
        words = random_words(rng, ct, nw, 3, 8) + random_words(rng, ct, nw // 2, 4, 8, force_equal_neighbour=True)
        for w in words:
            Wd = W(ct, 1, w)
            for I in Wd.subsets():
                # (A): every x in I and every one-pair configuration at I - x
                for x in I:
                    Ix = I - {x}
                    for (j, (a, b)) in one_pair_configs(Wd, Ix):
                        sp = transp(Wd.m, [(a, b)])
                        Jt = (Ix - {j, b}) | {a}
                        cnt = crossing_counts_concat(Wd, I, Ix, ident(Wd.m), Jt, sp)
                        twice = {k for k, v in cnt.items() if v >= 2}
                        if any(v > 2 for v in cnt.values()):
                            fails += 1; print("A >2", w, I, x, j, a, b)
                        if x == a:
                            kind = "A1"; exp = set()
                            # bijection: a -> b, b -> a, target I - {j,b}
                            tot = compose(ident(Wd.m), sp)
                            if not (Jt == I - {j, b} and tot[a] == b and tot[b] == a):
                                fails += 1; print("A1 target", w, I, x, j, a, b)
                        elif not (a < x < b):
                            kind = "A2"; exp = set()
                        else:
                            kind = "A3"; exp = {frozenset((x, a))}
                        if twice != exp:
                            fails += 1; print("A FAIL", kind, w, sorted(I), x, j, (a, b), twice)
                        # W never twice, mover never in a bigon
                        if any("W" in k for k in twice) or any(j in k for k in twice):
                            fails += 1
                        stats[kind] += 1
                        if kind == "A3":
                            cx, cp = Wd.c(x), Wd.c(a)
                            stats["A3-" + ("same" if cx == cp else "adj" if Wd.adj(cx, cp) else "far")] += 1
                # (B): every one-pair configuration at I and every x in its target J
                for (j, (a, b)) in one_pair_configs(Wd, I):
                    sp = transp(Wd.m, [(a, b)])
                    J = (I - {j, b}) | {a}
                    for x in J:
                        Kt = J - {x}
                        cnt = crossing_counts_concat(Wd, I, J, sp, Kt, ident(Wd.m))
                        twice = {k for k, v in cnt.items() if v >= 2}
                        if any(v > 2 for v in cnt.values()):
                            fails += 1
                        if x == a:
                            kind = "B1"; exp = set()
                        elif not (a < x < b):
                            kind = "B2"; exp = set()
                        else:
                            kind = "B3"; exp = {frozenset((x, b))}
                        if twice != exp:
                            fails += 1; print("B FAIL", kind, w, sorted(I), x, j, (a, b), twice)
                        if any("W" in k for k in twice) or any(j in k for k in twice):
                            fails += 1
                        stats[kind] += 1
                        # target and bijection for B2/B3 agree with A
                        if kind != "B1":
                            if Kt != (I - {x, j, b}) | {a}:
                                fails += 1
    print("Prop 5.3 (pure crossing numbers; types A2,A3,A4,D4,D5,E6): failures", fails)
    for k, v in sorted(stats.items()):
        print("   ", k, v)
    return fails


# ----------------------------------------------------------------------------- signs
def check_signs(nw, seed=2):
    """Lemma 5.1 against formula.py, and the sign identities (C1)-(C3) of Thm C recomputed
    from formula.py's own signs (not from the closed forms of the proof)."""
    rng = random.Random(seed)
    fails = 0
    cnt = Counter()
    for ct in TYPES:
        for w in random_words(rng, ct, nw, 3, 9):
            st = Stack(ct, w)
            Wd = W(ct, 1, w)
            for I in Wd.subsets():
                terms = st.terms_from(I)
                one = {}
                for (J, sigma, mono, sign) in terms:
                    moved = [s for s in Wd.slots if sigma[s - 1] != s]
                    if len(moved) == 2:
                        a, b = moved
                        (j,) = [s for s in I - J if s != b]
                        one[(j, (a, b))] = sign
                        if sign != (-1) ** (nI(I, j) + NI(I, a, b)):
                            fails += 1; print("L5.1", w, I, j, a, b)
                        cnt["one-pair terms"] += 1
                if set(one) != set(one_pair_configs(Wd, I)):
                    fails += 1; print("one-pair set", w, I)
                ONE = one
                # (C1): bad triples
                for i, j in itertools.combinations(sorted(I), 2):
                    if not Wd.adj(Wd.c(i), Wd.c(j)):
                        continue
                    eps = (-1) ** (nI(I, i) + nI(I, j))
                    for y in range(1, i):
                        if Wd.c(y) != Wd.c(j):
                            continue
                        if y in I:
                            sA = (-1) ** nI(I, y) * dict_one(st, I - {y}, i, (y, j))
                            s = sA
                        else:
                            J = (I - {i, j}) | {y}
                            sB = ONE[(i, (y, j))] * (-1) ** nI(J, y)
                            s = sB
                        if s != -eps:
                            fails += 1; print("C1", w, I, i, j, y)
                        cnt["bad triples"] += 1
                # (C2), (C3)
                for (j, (a, b)), sg in ONE.items():
                    for x in I - {j, b}:
                        sA = (-1) ** nI(I, x) * dict_one(st, I - {x}, j, (a, b))
                        J = (I - {j, b}) | {a}
                        sB = sg * (-1) ** nI(J, x)
                        if sA != -sB:
                            fails += 1; print("C2", w, I, x, j, a, b)
                        cnt["(x,j,p) pairs"] += 1
                        if a < x < b and Wd.adj(Wd.c(x), Wd.c(a)):
                            sAp = (-1) ** nI(I, j) * dict_one(st, I - {j}, x, (a, b))
                            sBp = ONE[(x, (a, b))] * (-1) ** nI(J2(I, x, a, b), j)
                            if not (sAp == -sA and sBp == -sAp):
                                fails += 1; print("C3", w, I, x, j, a, b)
                            cnt["four-term groups"] += 1
    print("signs (Lemma 5.1, C1-C3 recomputed from formula.py; A4, D4, D5, E6): failures", fails, dict(cnt))
    return fails


_cache = {}


def dict_one(st, I, j, p):
    key = (tuple(st.ct), tuple(st.col), I)
    if key not in _cache:
        d = {}
        for (J, sigma, mono, sign) in st.terms_from(I):
            moved = [s for s in range(1, st.m + 1) if sigma[s - 1] != s]
            if len(moved) == 2:
                a, b = moved
                (jj,) = [s for s in I - J if s != b]
                d[(jj, (a, b))] = sign
        _cache[key] = d
    return _cache[key][(j, p)]


def J2(I, x, a, b):
    return (I - {x, b}) | {a}


# ----------------------------------------------------------------------------- Prop 7.1
def check_defect(n, seed=3):
    rng = random.Random(seed)
    fails = 0
    for k in range(n):
        ct = rng.choice(TYPES + [["A", 2]])
        nodes = sorted({c for e in dynkin(ct) for c in e})
        L = rng.randint(2, 9)
        w = tuple(rng.choice(nodes) for _ in range(L))
        red = rng.choice(nodes)
        Wd = W(ct, red, w)
        I = frozenset(s for s in Wd.slots if rng.random() < 0.5)
        J = frozenset(s for s in Wd.slots if rng.random() < 0.5)
        sigma = rng.choice(list(itertools.islice(Wd.bijections(), 50)))
        direct = Wd.defect(I, J, sigma)
        # Prop 7.1 formula
        f = Counter()
        for u, v in itertools.combinations(Wd.slots, 2):     # u < v source slots
            if sigma[u] < sigma[v]:
                fv = -2 * (v not in I and sigma[v] in J)
            else:
                fv = 1 if (v in I and sigma[u] not in J) else -1
            f.update({k2: val * fv for k2, val in Wd.xdeg(Wd.c(u), Wd.c(v)).items()})
        for s in Wd.slots:
            if s not in I and sigma[s] in J:
                f.update({k2: -2 * val for k2, val in Wd.xdeg(Wd.c(s), "W").items()})
        f = Counter({k2: v for k2, v in f.items() if v})
        if f != direct:
            fails += 1
            print("defect FAIL", ct, red, w, I, J, sigma, direct, f)
    print("Prop 7.1 vs direct grading:", n, "random (type, word, red, I, J, sigma); failures", fails)
    return fails


# ----------------------------------------------------------------------------- Prop 7.4
def check_prop74(nw, seed=4):
    rng = random.Random(seed)
    fails = 0
    stats = Counter()
    for ct in TYPES + [["A", 3]]:
        nodes = sorted({c for e in dynkin(ct) for c in e})
        words = random_words(rng, ct, nw, 3, 7, maxmult=2)
        for w in words:
            for red in nodes:
                Wd = W(ct, red, w)
                bij = list(Wd.bijections())
                V = list(Wd.subsets())
                for I in V:
                    found = set()
                    for J in V:
                        h = len(J) - len(I)
                        if h not in (-1, 0):
                            continue
                        for sigma in bij:
                            mono_ok = [s for s in Wd.solve(Wd.defect(I, J, sigma)) if sum(s[0].values()) == 1]
                            if not mono_ok:
                                continue
                            monotone = not any(s not in I and sigma[s] in J for s in Wd.slots)
                            if not monotone:
                                stats["non-monotone first-order h=%d" % h] += 1
                                continue
                            if h == 0:
                                fails += 1; print("7.4(b) FAIL", ct, red, w, I, J, sigma, mono_ok)
                                continue
                            for (n, m, l, a) in mono_ok:
                                moved = sorted(s for s in Wd.slots if sigma[s] != s)
                                good = (len(moved) == 2 and not a and not l)
                                if good:
                                    u, v = moved
                                    (j,) = [s for s in I - J if s != v] if v in I else [None]
                                    good = (j is not None and (j, (u, v)) in one_pair_configs(Wd, I) and
                                            J == (I - {j, v}) | {u} and
                                            n == {Wd.c(u): 1} and m == {(Wd.c(u), Wd.c(j)): 1})
                                    found.add((j, (u, v)))
                                if not good:
                                    fails += 1; print("7.4(a) FAIL", ct, red, w, sorted(I), sorted(J), sigma, (n, m, l, a))
                                stats["monotone first-order h=-1"] += 1
                    if found != set(one_pair_configs(Wd, I)):
                        fails += 1; print("7.4(a) converse FAIL", ct, red, w, I)
    print("Prop 7.4 (mult <= 2; A3, A4, D4, D5, E6, every red colour, all I, J, sigma): failures", fails, dict(stats))
    return fails


# ----------------------------------------------------------------------------- configurations
def configurations(Wd, I, j, maxpairs=4):
    """all sets F of disjoint pairs (a<b, a notin I, b in I, b != j, equal colours)"""
    avail = [(a, b) for a in Wd.slots for b in Wd.slots
             if a < b and a not in I and b in I and b != j and Wd.c(a) == Wd.c(b)]
    for k in range(0, maxpairs + 1):
        for F in itertools.combinations(avail, k):
            used = [x for p in F for x in p]
            if len(set(used)) == len(used):
                yield F


def link_graph(Wd, j, F):
    links, blocks = [], 0
    for (a, b) in F:
        if a < j < b:
            if Wd.c(a) == Wd.c(j):
                blocks += 1
            elif Wd.adj(Wd.c(a), Wd.c(j)):
                links.append(("M", (a, b)))
    for p, q in itertools.combinations(F, 2):
        (a, b), (a2, b2) = sorted((p, q))
        if a < a2 < b < b2:
            if Wd.c(a) == Wd.c(a2):
                blocks += 1
            elif Wd.adj(Wd.c(a), Wd.c(a2)):
                links.append((p, q))
    return links, blocks


def connected(nodes, links):
    adj = defaultdict(set)
    for u, v in links:
        adj[u].add(v); adj[v].add(u)
    seen, st = {"M"}, ["M"]
    while st:
        u = st.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); st.append(v)
    return seen == set(nodes)


def cyc7():
    ct, red, w = ["A", 2], 2, (1, 2, 2, 1, 1, 2, 2)
    Wd = W(ct, red, w)
    cyc = set()
    for I in Wd.subsets():
        for j in I:
            for F in configurations(Wd, I, j, 3):
                links, blocks = link_graph(Wd, j, F)
                if blocks == 0 and connected(["M"] + list(F), links) and len(links) > len(F):
                    cyc.add((j, F))
    print("cyclic configurations of 1221122 (mover, pairs), over all vertices:", sorted(cyc))
    # Thm 9.1(a): dot-free homogeneous elements of hom. degree -1 by r-degree
    bij = list(Wd.bijections())
    V = list(Wd.subsets())
    byr = Counter()
    for I in V:
        for J in V:
            if len(J) != len(I) - 1:
                continue
            for sigma in bij:
                for (n, m, l, a) in Wd.solve(Wd.defect(I, J, sigma)):
                    if not a:
                        byr[sum(n.values())] += 1
    print("Thm 9.1(a) dot-free homogeneous basis elements of hom. degree -1, by r-degree:", sorted(byr.items()))


def gauge(word, red, ct, maxr=6):
    Wd = W(ct, red, word)
    bij = list(Wd.bijections())
    V = list(Wd.subsets())
    out = []
    for I in V:
        for J in V:
            if len(J) != len(I):
                continue
            for sigma in bij:
                if I == J and all(sigma[s] == s for s in Wd.slots):
                    continue
                for (n, m, l, a) in Wd.solve(Wd.defect(I, J, sigma)):
                    if 1 <= sum(n.values()) <= maxr:
                        out.append((sorted(I), sorted(J), {s: t for s, t in sigma.items() if s != t}, n, m, a))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prop53":
        check_prop53(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "signs":
        check_signs(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == "defect":
        check_defect(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    elif cmd == "prop74":
        check_prop74(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    elif cmd == "cyc7":
        cyc7()
    elif cmd == "gauge":
        word = tuple(int(c) for c in sys.argv[2]); red = int(sys.argv[3])
        ct = [sys.argv[4], int(sys.argv[5])] if len(sys.argv) > 5 else ["A", 2]
        res = gauge(word, red, ct)
        print("homogeneous basis families of hom. degree 0, r-degree >= 1 (dots allowed):", len(res))
        for r_ in res[:40]:
            print("   ", r_)


# ----------------------------------------------------------------------------- tau_y has no triple
def check_notriple(nw=40, seed=5):
    """Every inversion of tau_y (Thm B) joins one of the strands y, i, j to a strand outside
    {y, i, j}; hence the inversion graph has no triangle, no braid move applies to any reduced
    word of tau_y, and all reduced words of tau_y give the same element (Matsumoto)."""
    rng = random.Random(seed)
    n = fails = 0
    for ct in TYPES + [["A", 2], ["A", 3]]:
        for w in random_words(rng, ct, nw, 3, 9):
            Wd = W(ct, 1, w)
            for I in Wd.subsets():
                for i, j in itertools.combinations(sorted(I), 2):
                    if not Wd.adj(Wd.c(i), Wd.c(j)):
                        continue
                    for y in range(1, i):
                        if Wd.c(y) != Wd.c(j):
                            continue
                        tau = transp(Wd.m, [(y, j)])
                        inv = Wd.inversions(I, I - {i, j}, tau)
                        mov = {y, i, j}
                        n += 1
                        if any(len(set(p) & mov) != 1 for p in inv):
                            fails += 1
                        tri = any(frozenset((a, b)) in inv and frozenset((b, c)) in inv and frozenset((a, c)) in inv
                                  for a, b, c in itertools.combinations(set().union(*inv), 3)) if inv else False
                        fails += tri
    print("tau_y inversion graph bipartite between {y,i,j} and the rest, no triangle: %d cases, failures %d" % (n, fails))


if __name__ == "__main__" and sys.argv[1] == "notriple":
    check_notriple()
