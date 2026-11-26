"""
01_prop3.py   (pure Python:  python3 proofs/checks/01_prop3.py [N_RANDOM])

Independent re-check of Lemma 1, Proposition 3 and Corollary 3.1 of PROOFS.md.
Nothing from formula.py / degree_model.py is used: own Dynkin data, own grading group
(formal sums over the generators q, s_c (=sigma_c), d_c (=delta_c), e_{ab} (unordered edge),
w_c (= eps_{cW})), own enumeration of configurations, own reduced words (bubble sort of the
position permutation, crossing degrees read off letter by letter from the colours of the
strands that actually cross).

For random words (types A_2..A_4, D_4, D_5, E_6; colours arbitrary, repeats allowed) and
random (I, j, F) with F an ARBITRARY set of disjoint same-colour pairs, a_p not in I,
b_p in I, b_p != j (not only admissible ones), checks
   D(I) - D(J) - deg(beta)  ==  sum_p [c(p),c(p)] + sum_{a_p<j<b_p} 2[c(j),c(p)]
                                + sum_{p,q interlaced} 2[c(p),c(q)]                 (Prop. 3)
and, when (j, F) is admissible in the sense of NOTES.md section 2 (tree, no blocks),
   deg( prod_p r_{c(p)} t_{c(p), c(x_p)} )  ==  D(I) - D(J) - deg(beta)            (Cor. 3.1)
and for F = {} that the diagram is m_j(I) with degree d(j) (Lemma 1).
"""
import itertools
import random
import sys
from collections import Counter

EDGES = {
    ("A", 2): [(1, 2)], ("A", 3): [(1, 2), (2, 3)], ("A", 4): [(1, 2), (2, 3), (3, 4)],
    ("D", 4): [(1, 2), (2, 3), (2, 4)], ("D", 5): [(1, 2), (2, 3), (3, 4), (3, 5)],
    ("E", 6): [(1, 3), (3, 4), (4, 5), (5, 6), (2, 4)],
}


class Grading:
    def __init__(self, typ):
        self.E = {frozenset(e) for e in EDGES[typ]}
        self.rank = typ[1]

    def adj(self, a, b):
        return frozenset((a, b)) in self.E

    def bracket(self, a, b):
        """[a, b] = degree of a crossing of black strands of colours a, b."""
        if a == b:
            return Counter({"q": -2, "s%d" % a: 1})
        if self.adj(a, b):
            return Counter({"q": 1, "e%s" % "".join(map(str, sorted((a, b)))): 1})
        return Counter()

    def crossing(self, a, b, red):
        """degree of a crossing between strands of colours a, b ('W' = red strand of colour red)."""
        if a == "W" or b == "W":
            c = b if a == "W" else a
            return Counter({"q": 1, "w%d" % c: 1}) if c == red else Counter()
        return self.bracket(a, b)

    def r(self, c):
        return Counter({"s%d" % c: 1, "d%d" % c: 1})

    def t(self, a, b):
        return Counter({"e%s" % "".join(map(str, sorted((a, b)))): 2, "d%d" % a: -1})


def add(*cs):
    out = Counter()
    for c in cs:
        for k, v in c.items():
            out[k] += v
    return Counter({k: v for k, v in out.items() if v})


def scale(c, n):
    return Counter({k: n * v for k, v in c.items() if n * v})


def d_of(G, col, red, x):
    out = G.crossing(col[x - 1], "W", red)
    for i in range(1, x):
        out = add(out, G.bracket(col[i - 1], col[x - 1]))
    return out


def D_of(G, col, red, I):
    return add(*[d_of(G, col, red, x) for x in I]) if I else Counter()


def state(m, I):
    """list of strand labels by position: slots not in I decreasing, 'W', slots of I increasing."""
    return [s for s in range(m, 0, -1) if s not in I] + ["W"] + sorted(I)


def diagram_degree(G, col, red, I, J, sigma):
    """degree of a reduced diagram e_I -> e_J in which the strand of slot s ends in slot sigma[s]."""
    m = len(col)
    src = state(m, I)
    tgt = state(m, J)
    inv = {sigma[s]: s for s in sigma}
    # target position of each strand (identified by its source slot)
    tgt_pos = {("W" if x == "W" else inv[x]): p for p, x in enumerate(tgt)}
    cur = list(src)
    deg = Counter()
    n_cross = 0
    changed = True
    while changed:                                  # bubble sort = a reduced word
        changed = False
        for p in range(len(cur) - 1):
            if tgt_pos[cur[p]] > tgt_pos[cur[p + 1]]:
                a, b = cur[p], cur[p + 1]
                ca = "W" if a == "W" else col[a - 1]
                cb = "W" if b == "W" else col[b - 1]
                deg = add(deg, G.crossing(ca, cb, red))
                cur[p], cur[p + 1] = b, a
                n_cross += 1
                changed = True
    return deg, n_cross


def rhs_prop3(G, col, j, F):
    c = lambda s: col[s - 1]
    out = Counter()
    for (a, b) in F:
        out = add(out, G.bracket(c(a), c(a)))
        if a < j < b:
            out = add(out, scale(G.bracket(c(j), c(a)), 2))
    for (a, b), (a2, b2) in itertools.combinations(F, 2):
        if a < a2 < b < b2 or a2 < a < b2 < b:
            out = add(out, scale(G.bracket(c(a), c(a2)), 2))
    return out


def tree_monomial_degree(G, col, j, F):
    """None if (j, F) not admissible; else degree of prod r_{c(p)} t_{c(p), c(x_p)}."""
    c = lambda s: col[s - 1]
    nodes = ["M"] + list(F)
    nbr = {x: [] for x in nodes}
    for p in F:
        a, b = p
        if a < j < b:
            if c(a) == c(j):
                return None
            if G.adj(c(a), c(j)):
                nbr["M"].append((p, j))
                nbr[p].append(("M", None))
    for p, q in itertools.combinations(F, 2):
        (a, b), (a2, b2) = sorted((p, q))
        if a < a2 < b < b2:
            if c(a) == c(a2):
                return None
            if G.adj(c(a), c(a2)):
                nbr[(a, b)].append(((a2, b2), b))       # b lies in (a2, b2)
                nbr[(a2, b2)].append(((a, b), a2))      # a2 lies in (a, b)
    n_links = sum(len(v) for v in nbr.values()) // 2
    if n_links != len(F):
        return None
    parent = {}
    seen = {"M"}
    stack = ["M"]
    while stack:
        x = stack.pop()
        for y, conn in nbr[x]:
            if y not in seen:
                seen.add(y)
                parent[y] = conn
                stack.append(y)
    if len(seen) != len(nodes):
        return None
    out = Counter()
    for p, conn in parent.items():
        out = add(out, G.r(c(p[0])), G.t(c(p[0]), c(conn)))
    return out


def random_config(rng, m, col):
    for _ in range(100):
        I = {s for s in range(1, m + 1) if rng.random() < 0.5}
        if not I:
            continue
        j = rng.choice(sorted(I))
        pairs = [(a, b) for a in range(1, m + 1) for b in range(a + 1, m + 1)
                 if col[a - 1] == col[b - 1] and a not in I and b in I and b != j]
        rng.shuffle(pairs)
        F, used = [], set()
        for p in pairs:
            if p[0] not in used and p[1] not in used and rng.random() < 0.6:
                F.append(p)
                used |= set(p)
        return I, j, tuple(sorted(F))
    return None


def main(n_random=20000, seed=11):
    rng = random.Random(seed)
    n3 = n31 = nL1 = n2 = 0
    for k in range(n_random):
        typ = rng.choice(list(EDGES))
        G = Grading(typ)
        m = rng.randint(3, 11)
        col = [rng.randint(1, typ[1]) for _ in range(m)]
        red = rng.randint(1, typ[1])
        cfg = random_config(rng, m, col)
        if k % 2 == 0:                       # every other sample: insist on at least one pair
            for _ in range(20):
                if cfg is not None and cfg[2]:
                    break
                cfg = random_config(rng, m, col)
        if cfg is None:
            continue
        I, j, F = cfg
        J = (I - {j} - {b for a, b in F}) | {a for a, b in F}
        sigma = {s: s for s in range(1, m + 1)}
        for a, b in F:
            sigma[a], sigma[b] = b, a
        degb, ncross = diagram_degree(G, col, red, I, J, sigma)
        lhs = add(D_of(G, col, red, I), scale(D_of(G, col, red, J), -1), scale(degb, -1))
        assert lhs == rhs_prop3(G, col, j, F), (typ, col, I, j, F, lhs, rhs_prop3(G, col, j, F))
        n3 += 1
        n2 += len(F) >= 2
        if not F:
            assert degb == d_of(G, col, red, j)
            assert ncross == 1 + (j - 1)              # red strand and every slot < j
            nL1 += 1
        mono = tree_monomial_degree(G, col, j, F)
        if mono is not None:
            assert mono == lhs, (typ, col, I, j, F)
            n31 += 1
    print("Proposition 3 identity: %d random configurations OK (%d with F empty: Lemma 1 OK; %d with |F| >= 2); "
          "Corollary 3.1 tree monomial has the right degree: %d admissible configurations OK" % (n3, nL1, n2, n31))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
