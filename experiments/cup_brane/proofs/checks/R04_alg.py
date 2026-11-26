"""
Referee checks for proofs/04_first_order.md -- algebra part (sage -python).

Independent of 04_lib.py / 04_*.py: words are built by my own strand simulation (R04_comb.W,
bubble-sort reduced words, NOT the klrw basis braid) and multiplied in the KLRW algebra through
checks/alg.py; signs of the tree formula from formula.Stack (authoritative).

    sage -python proofs/checks/R04_alg.py thmB  [NWORDS] [SEED]   Theorem B entrywise (exact) + mod r^2
    sage -python proofs/checks/R04_alg.py thmC  [NWORDS] [SEED]   (delta_0+delta_1)^2 has no r-degree <= 1 part
    sage -python proofs/checks/R04_alg.py vals  [NWORDS] [SEED]   Cor. 5.4: composites mod r
    sage -python proofs/checks/R04_alg.py ex83                    Example 8.3 from scratch
    sage -python proofs/checks/R04_alg.py inj   [NWORDS] [SEED]   Lemma 8.5 (injectivity) as a rank computation
"""
import sys, os, itertools, random, time
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from alg import Alg  # noqa: E402
from R04_comb import W, ident, transp, compose, nI, NI, one_pair_configs, random_words, dynkin  # noqa: E402
from formula import Stack  # noqa: E402


class AW:
    def __init__(self, ct, red, word):
        self.Wd = W(ct, red, word)
        self.ct, self.red, self.word = list(ct), red, tuple(word)
        toks = [str(c) for c in word] + ["W%d" % red]
        self.A = Alg(list(ct), " ".join(toks))
        self.K, self.D = self.A.K, self.A.D
        self.names = list(self.D.variable_names())
        self.rpos = [i for i, n in enumerate(self.names) if n.startswith("r_")]
        self.ppos = [i for i, n in enumerate(self.names) if not n.startswith("x_")]
        self.st = Stack(list(ct), tuple(word))
        self._cache = {}

    def seq(self, I):
        return " ".join("W%d" % self.red if l == "W" else str(self.Wd.c(l)) for l in self.Wd.labels(I))

    def welem(self, I, word):
        return self.A.word(self.seq(I), tuple(word))

    def el(self, I, J, sigma):
        key = (frozenset(I), frozenset(J), tuple(sorted(sigma.items())))
        if key not in self._cache:
            w, _ = self.Wd.bubble_word(frozenset(I), frozenset(J), sigma)
            self._cache[key] = self.welem(I, w)
        return self._cache[key]

    def move(self, I, x):
        I = frozenset(I)
        return self.el(I, I - {x}, ident(self.Wd.m))

    def r(self, c):
        return self.D("r_%d" % c)

    def t(self, c, c2):
        return self.D("t_%d_%d" % (c, c2))

    def mono(self, mono):
        e = self.D.one()
        for nm, p in mono:
            e *= self.D(nm) ** p
        return e

    def dotvar(self, I, slot):
        c = self.Wd.c(slot)
        k = 0
        for lab in self.Wd.labels(I):
            if lab != "W" and self.Wd.c(lab) == c:
                k += 1
                if lab == slot:
                    return self.D("x_%d_%d" % (c, k))
        raise ValueError

    # -------------------------------------------------------------- r-degree parts
    def rdeg(self, e):
        return sum(e[p] for p in self.rpos)

    def part(self, el, keep):
        res = self.K.zero()
        for braid, coeff in el:
            d = {e: c for e, c in coeff.dict().items() if keep(e)}
            if d:
                res += self.K.term(braid, self.D(d))
        return res

    def rpart_le(self, el, k):
        return self.part(el, lambda e: self.rdeg(e) <= k)

    def rpart_eq(self, el, k):
        return self.part(el, lambda e: self.rdeg(e) == k)

    def tree_terms(self, I, maxpairs):
        out = []
        for (J, sigma, mono, sign) in self.st.terms_from(frozenset(I)):
            k = sum(p for nm, p in mono if nm.startswith("r_"))
            if k <= maxpairs:
                out.append((J, {s: sigma[s - 1] for s in range(1, len(sigma) + 1)}, mono, sign, k))
        return out


# ------------------------------------------------------------------ Lemma 4.5 word
def slide(lab, u, targets):
    """strand u moves left past the strands `targets` in that order, each its left neighbour."""
    letters = []
    for v in targets:
        p = lab.index(u)
        assert p > 0 and lab[p - 1] == v, (lab, u, v)
        lab[p - 1], lab[p] = lab[p], lab[p - 1]
        letters.append(p)            # 1-based letter of the crossing of positions p, p+1 (= p-1, p 0-based)
    return letters


def X_word(Wd, I, i, j, y):
    """The word X_y of Lemma 4.5, built from its verbal description."""
    ZR = sorted([z for z in I if z < i], reverse=True)
    ZL = sorted([z for z in Wd.slots if z not in I and z < i])
    z = ZR + ["W"] + ZL
    UR = sorted([u for u in I if i < u < j], reverse=True)
    UL = sorted([u for u in Wd.slots if u not in I and i < u < j])
    s = z.index(y)            # 0-based; y = z_{s+1} in 1-based
    lab = Wd.labels(I)
    w = []
    w += slide(lab, i, z[:s])
    w += slide(lab, j, UR + z[:s])
    # now (y, i, j) consecutive
    p = lab.index(y)
    assert lab[p + 1] == i and lab[p + 2] == j
    w += slide(lab, y, z[s + 1:])
    w += slide(lab, i, z[s + 1:])
    w += slide(lab, y, UL)
    return tuple(w), lab


def thmB(nw, seed):
    rng = random.Random(seed)
    fails = 0
    stats = Counter()
    t0 = time.time()
    words = []
    for ct in (["A", 4], ["D", 4], ["D", 5], ["E", 6], ["A", 2]):
        nodes = sorted({c for e in dynkin(ct) for c in e})
        ws = random_words(rng, ct, nw, 4, 7)
        # words with three/four equal colours and equal neighbours
        ws += [w for w in random_words(rng, ct, 4 * nw, 5, 8) if max(Counter(w).values()) >= 3][:nw]
        for w in ws:
            words.append((ct, rng.choice(nodes), w))
    # hand-picked: D4 trivalent node 2, E6 trivalent node 4
    words += [(["D", 4], 2, (2, 1, 3, 4, 2, 2, 1)), (["D", 4], 1, (1, 2, 1, 3, 2, 4, 2)),
              (["E", 6], 4, (4, 2, 3, 5, 4, 4, 2)), (["E", 6], 2, (2, 4, 2, 3, 4, 5, 4)),
              (["A", 2], 1, (1, 2, 2, 1, 1, 2, 2)), (["A", 2], 2, (1, 1, 2, 1, 1, 2, 1))]
    for ct, red, w in words:
        C = AW(ct, red, w)
        Wd = C.Wd
        V = [I for I in Wd.subsets() if len(I) >= 2]
        if len(V) > 40:
            V = rng.sample(V, 40)
        for I in V:
            sq = defaultdict(lambda: C.K.zero())
            for x1 in I:
                s1 = (-1) ** nI(I, x1)
                for x2 in I - {x1}:
                    s2 = (-1) ** nI(I - {x1}, x2)
                    sq[I - {x1, x2}] += C.D(s1 * s2) * (C.move(I, x1) * C.move(I - {x1}, x2))
            for i, j in itertools.combinations(sorted(I), 2):
                K = I - {i, j}
                comp = sq.get(K, C.K.zero())
                eps = (-1) ** (nI(I, i) + nI(I, j))
                bad = [y for y in range(1, i) if Wd.c(y) == Wd.c(j) and Wd.adj(Wd.c(i), Wd.c(j))]
                predX = C.K.zero()
                predB = C.K.zero()
                for y in bad:
                    wX, lab = X_word(Wd, I, i, j, y)
                    tau = transp(Wd.m, [(y, j)])
                    X = C.welem(I, wX)
                    # bijection of X_y is tau_y (check by labels)
                    exp_lab = Wd.labels(K)
                    got = [l if l == "W" else tau[l] for l in lab]
                    if got != exp_lab:
                        fails += 1; print("tau FAIL", ct, red, w, I, i, j, y)
                    if len(wX) != len(Wd.inversions(I, K, tau)):
                        fails += 1; print("X not reduced", ct, red, w, I, i, j, y)
                    Bm = C.el(I, K, tau)
                    predX += X
                    predB += Bm
                    stats["bad triples"] += 1
                    stats["X_y == my bubble word exactly" if X == Bm else "X_y != my bubble word"] += 1
                coef = C.D(eps) * C.r(Wd.c(j)) * C.t(Wd.c(j), Wd.c(i)) if bad else C.D(0)
                if comp != coef * predX:
                    fails += 1; print("Thm B exact FAIL", ct, red, w, sorted(I), i, j, bad)
                if not C.rpart_le(comp - coef * predB, 1).is_zero():
                    fails += 1; print("Thm B mod r^2 FAIL", ct, red, w, sorted(I), i, j, bad)
                if bad and not C.rpart_le(comp, 1) != 0:
                    pass
                stats["components"] += 1
                stats["nonzero components" if bad else "zero components"] += 1
    print("Theorem B (exact, with X_y built from Lemma 4.5; and mod r^2 with my own reduced word):",
          "words", len(words), "failures", fails, dict(stats), "[%.0fs]" % (time.time() - t0))
    return fails


def thmC(nw, seed, flip=False, maxv=30):
    rng = random.Random(seed)
    fails = 0
    stats = Counter()
    t0 = time.time()
    words = []
    for ct in (["A", 4], ["D", 4], ["D", 5], ["E", 6], ["A", 2], ["A", 3]):
        nodes = sorted({c for e in dynkin(ct) for c in e})
        ws = random_words(rng, ct, nw, 4, 7)
        ws += [w for w in random_words(rng, ct, 4 * nw, 5, 8) if max(Counter(w).values()) >= 3][:nw]
        for w in ws:
            words.append((ct, rng.choice(nodes), w))
    words += [(["D", 4], 2, (2, 1, 3, 4, 2, 2, 1)), (["D", 4], 1, (1, 2, 1, 3, 2, 4, 2)),
              (["E", 6], 4, (4, 2, 3, 5, 4, 4, 2)), (["E", 6], 2, (2, 4, 2, 3, 4, 5, 4)),
              (["A", 2], 2, (1, 2, 2, 1, 1, 2, 2)), (["A", 2], 1, (1, 2, 2, 1, 1, 2, 2)),
              (["A", 3], 2, (1, 2, 3, 2, 1, 2, 2, 3)), (["D", 4], 4, (2, 4, 2, 2, 4, 2))]
    for ct, red, w in words:
        C = AW(ct, red, w)
        Wd = C.Wd
        V = [I for I in Wd.subsets() if len(I) >= 2]
        if len(V) > maxv:
            V = rng.sample(V, maxv)
        cache = {}

        def terms(I):
            if I not in cache:
                lst = []
                for (J, sig, mono, sign, k) in C.tree_terms(I, 1):
                    if flip and k == 1:
                        sign = -sign
                    lst.append((J, (C.D(sign) * C.mono(mono)) * C.el(I, J, sig), k))
                cache[I] = lst
            return cache[I]
        wf = 0
        for I in V:
            tot = defaultdict(lambda: C.K.zero())
            for (J, e1, k1) in terms(I):
                for (K, e2, k2) in terms(J):
                    tot[K] += e1 * e2
                    stats["products"] += 1
            for K, v in tot.items():
                if not C.rpart_le(v, 1).is_zero():
                    wf += 1
                    if not flip:
                        print("Thm C FAIL", ct, red, w, sorted(I), sorted(K), C.rpart_le(v, 1))
            stats["vertices"] += 1
        fails += bool(wf)
        stats["words with delta0^2 != 0 mod r^2"] += any(
            Wd.c(y) == Wd.c(j) and Wd.adj(Wd.c(i), Wd.c(j)) for y, i, j in itertools.combinations(Wd.slots, 3))
    print("Theorem C%s: words %d, words with a failing vertex %d, %s [%.0fs]" % (
        " (NEGATIVE CONTROL: one-pair signs flipped)" if flip else "", len(words), fails, dict(stats), time.time() - t0))
    return fails


def vals(nw, seed, maxv=25):
    """Cor. 5.4: every composite (A) / (B), mod r, against the table."""
    rng = random.Random(seed)
    fails = 0
    stats = Counter()
    t0 = time.time()
    for ct in (["A", 4], ["D", 4], ["D", 5], ["E", 6], ["A", 2]):
        nodes = sorted({c for e in dynkin(ct) for c in e})
        for w in random_words(rng, ct, nw, 4, 7):
            red = rng.choice(nodes)
            C = AW(ct, red, w)
            Wd = C.Wd
            V = list(Wd.subsets())
            if len(V) > maxv:
                V = rng.sample(V, maxv)
            for I in V:
                for x in I:
                    Ix = I - {x}
                    for (j, (a, b)) in one_pair_configs(Wd, Ix):
                        sp = transp(Wd.m, [(a, b)])
                        Jt = (Ix - {j, b}) | {a}
                        prod = C.move(I, x) * C.el(Ix, Jt, sp)
                        pred = predicted(C, I, Jt, x, j, a, b, "A")
                        stats["A"] += 1
                        if C.rpart_le(prod, 0) != C.rpart_le(pred, 0):
                            fails += 1; print("vals A FAIL", ct, red, w, sorted(I), x, j, (a, b))
                for (j, (a, b)) in one_pair_configs(Wd, I):
                    sp = transp(Wd.m, [(a, b)])
                    J = (I - {j, b}) | {a}
                    for x in J:
                        prod = C.el(I, J, sp) * C.move(J, x)
                        pred = predicted(C, I, J - {x}, x, j, a, b, "B")
                        stats["B"] += 1
                        if C.rpart_le(prod, 0) != C.rpart_le(pred, 0):
                            fails += 1; print("vals B FAIL", ct, red, w, sorted(I), x, j, (a, b))
    print("Cor. 5.4 (composites mod r; A4, D4, D5, E6, A2): failures", fails, dict(stats), "[%.0fs]" % (time.time() - t0))
    return fails


def predicted(C, I, K, x, j, a, b, side):
    Wd = C.Wd
    if x == a:
        sig = transp(Wd.m, [(a, b)])
        return C.el(I, K, sig)
    sig = transp(Wd.m, [(a, b)])
    base = C.el(I, K, sig)
    if not (a < x < b):
        return base
    cx, cp = Wd.c(x), Wd.c(a)
    if cx == cp:
        return C.K.zero()
    if not Wd.adj(cx, cp):
        return base
    other = a if side == "A" else b
    Q = C.t(cx, cp) * C.dotvar(I, x) + C.t(cp, cx) * C.dotvar(I, other)
    return Q * base


# ------------------------------------------------------------------ maps on the cube
def full_d(C, maxpairs=99):
    d = {}
    for I in C.Wd.subsets():
        row = defaultdict(lambda: C.K.zero())
        for (J, sig, mono, sign, k) in C.tree_terms(I, maxpairs):
            row[J] += (C.D(sign) * C.mono(mono)) * C.el(I, J, sig)
        d[I] = dict(row)
    return d


def mcompose(C, f, g):
    res = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for I, row in f.items():
        for J, x in row.items():
            for K, y in g.get(J, {}).items():
                res[I][K] += x * y
    return res


def madd(C, *terms):
    res = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for c, mp in terms:
        for I, row in mp.items():
            for J, x in row.items():
                res[I][J] += C.D(c) * x
    return res


def mzero(mp, f=None):
    return all((f(x) if f else x).is_zero() for row in mp.values() for x in row.values())


def ex83():
    C = AW(["A", 2], 2, (1, 2, 1, 2))
    Wd = C.Wd
    V = list(Wd.subsets())
    d = full_d(C)
    print("1212, red 2: tree formula d^2 = 0 exactly (own reduced words):", mzero(mcompose(C, d, d)))
    I0, J0 = frozenset({3, 4}), frozenset({1, 2})
    sig = transp(4, [(2, 4)])
    wd, _ = Wd.bubble_word(I0, J0, sig)
    print("h: word", wd, "from state", C.seq(I0), "to", C.seq(J0))
    h = {I0: {J0: C.r(2) * C.t(2, 1) * C.el(I0, J0, sig)}}
    print("   h^2 = 0:", mzero(mcompose(C, h, h)))
    one = {I: {I: C.A.e(C.seq(I))} for I in V}
    g = madd(C, (1, one), (1, h))
    gi = madd(C, (1, one), (-1, h))
    print("   g g^-1 = 1:", mzero(madd(C, (1, mcompose(C, g, gi)), (-1, one))))
    dp = mcompose(C, mcompose(C, g, d), gi)
    print("   d'^2 = 0:", mzero(mcompose(C, dp, dp)))
    d0 = {I: {J: x for J, x in row.items()} for I, row in full_d(C, 0).items()}
    diff0 = madd(C, (1, dp), (-1, d0))
    print("   (H1) d' - delta_0 has no r-free coefficient:", mzero(diff0, lambda x: C.rpart_le(x, 0)))
    diff = madd(C, (1, dp), (-1, d))
    print("   d' - d components with non-zero part of r-degree <= 1:")
    for I, row in sorted(diff.items(), key=lambda kv: sorted(kv[0])):
        for J, x in row.items():
            y = C.rpart_le(x, 1)
            if not y.is_zero():
                print("      %s -> %s : %s" % (sorted(I), sorted(J), y))
                for braid, coeff in y:
                    lab = Wd.labels(I)
                    for letter in braid.word():
                        lab[letter - 1], lab[letter] = lab[letter], lab[letter - 1]
                    tl = Wd.labels(J)
                    sigma = {a: b for a, b in zip(lab, tl) if a != "W"}
                    LR = [s for s in Wd.slots if s not in I and sigma[s] in J]
                    print("          bijection", {s: t for s, t in sigma.items() if s != t},
                          " strands crossing W left->right:", LR, " dots:", any(
                              e[p] for e in coeff.dict() for p in range(len(C.names)) if C.names[p].startswith("x_")))
    print("   full d' - d (all orders):")
    for I, row in sorted(diff.items(), key=lambda kv: sorted(kv[0])):
        for J, x in row.items():
            if not x.is_zero():
                print("      %s -> %s : %s" % (sorted(I), sorted(J), x))
    # the tree terms in the changed components
    for I in (frozenset({3, 4}),):
        for J, x in d[I].items():
            print("   d (tree formula) %s -> %s : %s" % (sorted(I), sorted(J), x))


def inj(nw, seed):
    """Lemma 8.5: eta -> [delta_0, eta] mod r^2 is injective on the span of one-pair terms."""
    from sage.all import matrix, QQ
    rng = random.Random(seed)
    t0 = time.time()
    words = [(["A", 2], 2, (1, 2, 1, 2)), (["A", 2], 1, (1, 1, 2, 1, 1)), (["A", 2], 2, (1, 2, 2, 1, 2)),
             (["D", 4], 2, (2, 1, 3, 2, 4, 2)), (["A", 3], 1, (1, 2, 3, 2, 1, 2))]
    for ct in (["A", 3], ["D", 4]):
        nodes = sorted({c for e in dynkin(ct) for c in e})
        for w in random_words(rng, ct, nw, 4, 6):
            words.append((ct, rng.choice(nodes), w))
    allok = True
    for ct, red, w in words:
        C = AW(ct, red, w)
        Wd = C.Wd
        V = list(Wd.subsets())
        d0 = full_d(C, 0)
        cols = []
        idx = {}
        for I in V:
            for (j, (a, b)) in one_pair_configs(Wd, I):
                J = (I - {j, b}) | {a}
                el = C.r(Wd.c(a)) * C.t(Wd.c(a), Wd.c(j)) * C.el(I, J, transp(Wd.m, [(a, b)]))
                eta = {I: {J: el}}
                com = madd(C, (1, mcompose(C, d0, eta)), (1, mcompose(C, eta, d0)))
                vec = {}
                for I2, row in com.items():
                    for K2, x in row.items():
                        for braid, coeff in C.rpart_eq(x, 1):
                            for e, c in coeff.dict().items():
                                key = (I2, K2, braid, e)
                                if key not in idx:
                                    idx[key] = len(idx)
                                vec[idx[key]] = vec.get(idx[key], 0) + int(c)
                cols.append(vec)
        if not cols:
            print("   ", ct, red, w, "no one-pair terms"); continue
        M = matrix(QQ, len(idx), len(cols), {(r_, c_): v for c_, vec in enumerate(cols) for r_, v in vec.items()})
        rk = M.rank()
        ok = rk == len(cols)
        allok &= ok
        print("    %s red %d word %s: one-pair terms %d, rank of [delta_0, -] mod r^2: %d %s" % (
            "".join(map(str, ct)), red, "".join(map(str, w)), len(cols), rk, "OK" if ok else "FAIL"))
    print("Lemma 8.5 injectivity:", "all OK" if allok else "FAILURES", "[%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    cmd = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if cmd == "thmB":
        thmB(nw, seed)
    elif cmd == "thmC":
        thmC(nw, seed)
        thmC(max(1, nw // 2), seed + 1, flip=True, maxv=10)
    elif cmd == "vals":
        vals(nw, seed)
    elif cmd == "ex83":
        ex83()
    elif cmd == "inj":
        inj(nw, seed)
