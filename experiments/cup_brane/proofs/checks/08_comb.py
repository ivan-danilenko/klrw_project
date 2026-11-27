"""
08 -- pure-Python combinatorics for the cyclic normal form (proofs/08_cyclic_normal_form.md).

No sage.  Import from another script in this folder with

    import importlib, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    C8 = importlib.import_module("08_comb")

Conventions (README_FOR_PROVERS, PROOFS.md §1, 04_first_order.md §7):
  * slots 1..m, colour c(s) = word[s-1]; a vertex I is a frozenset of slots (the slots whose
    strand is to the right of the red strand W of colour `red`);
  * position of slot s at the vertex I:  +s if s in I, -s otherwise; W at 0;
  * a bijection sigma is a tuple, sigma[s-1] = the target slot of the strand starting in slot s;
    it is colour preserving;
  * a "term" (I, J, sigma, mono) is the klrw basis braid of sigma from e_I to e_J times the
    parameter monomial mono = tuple(sorted((name, power))) with names r_c, t_c_d (u never occurs);
  * crossing content (NOTES §3): ('s', c) for a crossing of two strands of colour c,
    ('e', c, d) (c < d) for adjacent colours, ('w',) for a strand of the red colour crossing W.
    r_c t_{cd} has content s_c + 2 e_{cd}.

Main entry points:
    W = WordData(ct, red, word)
    W.dotfree_monomial(I, J, sigma)      -> mono or None  (dot-free homogeneous of degree D(I)-D(J))
    W.candidates(h, sources, rmin, rmax) -> list of (I, J, sigma, mono), all dot-free homogeneous
                                            basis elements of homological degree h (|J| = |I| + h),
                                            found by a DFS over sigma pruned by the crossing content
    W.tree_terms(I)                      -> formula.Stack terms (J, sigma, mono, sign)
"""
import itertools
import os
import sys
import importlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from formula import Stack  # noqa: E402  (authoritative tree formula and signs)

L3 = importlib.import_module("03_lib")


def dynkin_edges(ct):
    t, n = ct
    if t == "A":
        E = [(i, i + 1) for i in range(1, n)]
    elif t == "D":
        E = [(i, i + 1) for i in range(1, n - 1)] + [(n - 2, n)]
    elif t == "E":
        E = [(1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)][: n - 2] + [(2, 4)]
    else:
        raise ValueError(t)
    return set(frozenset(e) for e in E)


def mono_str(mono):
    return "*".join(n if p == 1 else "%s^%d" % (n, p) for n, p in mono) or "1"


def rdeg(mono):
    return sum(p for n, p in mono if n.startswith("r_"))


class WordData:
    def __init__(self, ct, red, word):
        self.ct = [ct[0], int(ct[1])]
        self.red = int(red)
        self.col = tuple(int(c) for c in word)
        self.m = len(self.col)
        self.slots = tuple(range(1, self.m + 1))
        self.E = dynkin_edges(self.ct)
        self.nodes = sorted({c for e in self.E for c in e} | set(self.col) | {self.red})
        self.nbrs = {c: sorted(d for d in self.nodes if frozenset((c, d)) in self.E) for c in self.nodes}
        self.classes = defaultdict(list)
        for s in self.slots:
            self.classes[self.c(s)].append(s)
        self.stack = Stack(self.ct, self.col)
        self._D = {}

    # ------------------------------------------------------------------ basics
    def c(self, s):
        return self.col[s - 1]

    def adj(self, a, b):
        return frozenset((a, b)) in self.E

    def bracket(self, c1, c2):
        if c1 == c2:
            return ("s", c1)
        if self.adj(c1, c2):
            return ("e", min(c1, c2), max(c1, c2))
        return None

    @staticmethod
    def pos(I, s):
        return s if s in I else -s

    def labels(self, I):
        """left-to-right strand labels at the vertex I ('W' = red strand)."""
        return [s for s in sorted(set(self.slots) - set(I), reverse=True)] + ["W"] + sorted(I)

    def D(self, I):
        """crossing content of the geometric path I -> empty set (Lemma 7.0 of 04)."""
        I = frozenset(I)
        if I not in self._D:
            d = Counter()
            for v in I:
                if self.c(v) == self.red:
                    d[("w",)] += 1
                for u in range(1, v):
                    b = self.bracket(self.c(u), self.c(v))
                    if b:
                        d[b] += 1
            self._D[I] = d
        return self._D[I]

    def content(self, I, J, sigma):
        """crossing content of the reduced diagram of sigma from e_I to e_J."""
        cnt = Counter()
        for s in self.slots:
            a, b = self.pos(I, s), self.pos(J, sigma[s - 1])
            if self.c(s) == self.red and a * b < 0:
                cnt[("w",)] += 1
            for t in range(s + 1, self.m + 1):
                if (a - self.pos(I, t)) * (b - self.pos(J, sigma[t - 1])) < 0:
                    br = self.bracket(self.c(s), self.c(t))
                    if br:
                        cnt[br] += 1
        return cnt

    def deficit(self, I, J, sigma):
        d = Counter(self.D(I))
        d.subtract(self.D(J))
        d.subtract(self.content(I, J, sigma))
        return {k: v for k, v in d.items() if v}

    # ------------------------------------------------------------------ monomials
    def solve_monomial(self, deficit, dots_allowed=False):
        """All monomials prod r^n t^m (and the dot numbers a_c) of the given crossing content.
        Dot-free: a unique monomial or none (forest argument).  Returns list of (mono, a)."""
        if any(v < 0 for v in deficit.values()):
            return []
        if deficit.get(("w",), 0):
            return []          # would need u (never homogeneous with dots >= 0 either: Cor 7.2 of 04)
        n = {c: deficit.get(("s", c), 0) for c in self.nodes}
        half = {}
        for e in self.E:
            a, b = sorted(e)
            v = deficit.get(("e", a, b), 0)
            if v % 2:
                return []
            half[a, b] = v // 2
        for k in deficit:
            if k[0] == "s" and k[1] not in self.nodes:
                return []
        edges = sorted(half)
        out = []
        if not dots_allowed:
            # leaf elimination on the forest
            need = dict(n)
            m = {}
            alive = {c: set(self.nbrs[c]) for c in self.nodes}
            done = set()
            changed = True
            while changed:
                changed = False
                for c in self.nodes:
                    if c in done or len(alive[c]) != 1:
                        continue
                    (d,) = alive[c]
                    e = (min(c, d), max(c, d))
                    m[c, d] = need[c]
                    m[d, c] = half[e] - need[c]
                    if m[d, c] < 0:
                        return []
                    need[d] -= m[d, c]
                    alive[d].discard(c)
                    alive[c] = set()
                    done.add(c)
                    changed = True
            for c in self.nodes:
                if c not in done and need[c] != 0:
                    return []
                if need[c] < 0:
                    return []
            mono = Counter()
            for c, v in n.items():
                if v:
                    mono["r_%d" % c] += v
            for (c, d), v in m.items():
                if v < 0:
                    return []
                if v:
                    mono["t_%d_%d" % (c, d)] += v
            return [(tuple(sorted(mono.items())), {})]
        # with dots: enumerate splits
        for split in itertools.product(*[range(half[e] + 1) for e in edges]):
            m = {}
            for (a, b), k in zip(edges, split):
                m[a, b] = k
                m[b, a] = half[a, b] - k
            a_ = {c: sum(m.get((c, d), 0) for d in self.nbrs[c]) - n[c] for c in self.nodes}
            if any(v < 0 for v in a_.values()):
                continue
            if any(v > 0 and c not in self.classes for c, v in a_.items()):
                continue
            mono = Counter()
            for c, v in n.items():
                if v:
                    mono["r_%d" % c] += v
            for (c, d), v in m.items():
                if v:
                    mono["t_%d_%d" % (c, d)] += v
            out.append((tuple(sorted(mono.items())), {c: v for c, v in a_.items() if v}))
        return out

    def dotfree_monomial(self, I, J, sigma):
        res = self.solve_monomial(self.deficit(I, J, sigma))
        return res[0][0] if res else None

    # ------------------------------------------------------------------ enumeration
    def _dfs_sigma(self, I, J, target, rmin):
        """DFS over colour-preserving sigma; prune on the crossing content."""
        order = list(self.slots)
        m = self.m
        posI = {s: self.pos(I, s) for s in self.slots}
        posJ = {s: self.pos(J, s) for s in self.slots}
        br = {}
        for s in self.slots:
            for t in self.slots:
                if s < t:
                    br[s, t] = self.bracket(self.c(s), self.c(t))
        sig = {}
        used = set()
        deficit = dict(target)
        out = []

        def srest():
            return sum(v for k, v in deficit.items() if k[0] == "s")

        def rec(k):
            if k == m:
                out.append(tuple(sig[s] for s in self.slots))
                return
            s = order[k]
            for tgt in self.classes[self.c(s)]:
                if tgt in used:
                    continue
                changes = []
                ok = True
                a, b = posI[s], posJ[tgt]
                if self.c(s) == self.red and a * b < 0:
                    changes.append(("w",))
                for t in order[:k]:
                    bb = br[min(s, t), max(s, t)]
                    if bb is None:
                        continue
                    if (a - posI[t]) * (b - posJ[sig[t]]) < 0:
                        changes.append(bb)
                for ch in changes:
                    v = deficit.get(ch, 0) - 1
                    deficit[ch] = v
                    if v < 0:
                        ok = False
                if ok and rmin and srest() < rmin:
                    ok = False
                if ok:
                    sig[s] = tgt
                    used.add(tgt)
                    rec(k + 1)
                    used.discard(tgt)
                    del sig[s]
                for ch in changes:
                    deficit[ch] += 1

        rec(0)
        return out

    def candidates(self, h=-1, sources=None, rmin=0, rmax=None, monotone=False, dots=False):
        """All dot-free homogeneous basis elements (I, J, sigma, mono) of homological degree h.
        With dots=True: (I, J, sigma, mono, a) for all homogeneous basis elements (a = dot numbers per colour)."""
        if sources is None:
            sources = [frozenset(c) for k in range(self.m + 1) for c in itertools.combinations(self.slots, k)]
        out = []
        for I in sources:
            I = frozenset(I)
            size = len(I) + h
            if size < 0 or size > self.m:
                continue
            for Jt in itertools.combinations(self.slots, size):
                J = frozenset(Jt)
                target = Counter(self.D(I))
                target.subtract(self.D(J))
                if any(v < 0 for v in target.values()):
                    continue
                target = {k: v for k, v in target.items() if v}
                for sigma in self._dfs_sigma(I, J, target, rmin):
                    if monotone and any(s not in I and sigma[s - 1] in J for s in self.slots):
                        continue
                    d = self.deficit(I, J, sigma)
                    for mono, a in self.solve_monomial(d, dots_allowed=dots):
                        rd = rdeg(mono)
                        if rd < rmin or (rmax is not None and rd > rmax):
                            continue
                        if dots:
                            out.append((I, J, sigma, mono, a))
                        else:
                            out.append((I, J, sigma, mono))
        return out

    # ------------------------------------------------------------------ reduced words
    def reduced_word(self, I, J, sigma, mode="left"):
        """A reduced word (1-based letters, first letter acts first on the source state) of the
        bijection sigma from e_I to e_J: repeatedly cross the leftmost ('left') or rightmost
        ('right') adjacent pair of strands that is inverted."""
        lab = self.labels(I)
        tgt = {s: self.pos(J, sigma[s - 1]) for s in self.slots}
        tgt["W"] = 0
        word = []
        n = len(lab)
        while True:
            rng = range(n - 1) if mode == "left" else range(n - 2, -1, -1)
            for p in rng:
                if tgt[lab[p]] > tgt[lab[p + 1]]:
                    lab[p], lab[p + 1] = lab[p + 1], lab[p]
                    word.append(p + 1)
                    break
            else:
                break
        return tuple(word)

    def sigma_from_word(self, I, J, word):
        """bijection of a braid word (1-based letters, first letter first) from e_I to e_J."""
        lab = self.labels(I)
        for p in word:
            lab[p - 1], lab[p] = lab[p], lab[p - 1]
        tl = self.labels(J)
        sig = {}
        for a, b in zip(lab, tl):
            if a == "W":
                assert b == "W"
                continue
            sig[a] = b
        return tuple(sig[s] for s in self.slots)

    # ------------------------------------------------------------------ the tree formula
    def tree_terms(self, I):
        return self.stack.terms_from(frozenset(I))

    def all_tree_terms(self):
        out = {}
        for k in range(1, self.m + 1):
            for c in itertools.combinations(self.slots, k):
                I = frozenset(c)
                for J, sigma, mono, sign in self.tree_terms(I):
                    out[(I, J, sigma, mono)] = sign
        return out

    # ------------------------------------------------------------------ descriptions
    def describe(self, I, J, sigma):
        """'mover/handover' description: which strands cross W, how the left/right strands move."""
        LR = [s for s in self.slots if s not in I and sigma[s - 1] in J]
        RL = [s for s in self.slots if s in I and sigma[s - 1] not in J]
        R = ["%d>%d" % (s, sigma[s - 1]) for s in sorted(I) if sigma[s - 1] in J and sigma[s - 1] != s]
        Lf = ["%d>%d" % (s, sigma[s - 1]) for s in sorted(set(self.slots) - set(I)) if sigma[s - 1] not in J and sigma[s - 1] != s]
        cross = ["%d>%d" % (s, sigma[s - 1]) for s in RL] + ["LR %d>%d" % (s, sigma[s - 1]) for s in LR]
        return "W: %s | R: %s | L: %s" % (" ".join(cross) or "-", " ".join(R) or "-", " ".join(Lf) or "-")

    def cycles(self, sigma):
        seen = set()
        out = []
        for s in self.slots:
            if s in seen or sigma[s - 1] == s:
                continue
            cyc = [s]
            seen.add(s)
            x = sigma[s - 1]
            while x != s:
                cyc.append(x)
                seen.add(x)
                x = sigma[x - 1]
            out.append(tuple(cyc))
        return out

    def cycle_str(self, sigma):
        return "".join("(" + " ".join(map(str, c)) + ")" for c in self.cycles(sigma)) or "id"

    # ------------------------------------------------------------------ cyclicity (03)
    def is_cyclic(self):
        Wd = L3.Word(list(self.col), lambda a, b: self.adj(a, b))
        return Wd.search() is not None

    def minimal_cyclic(self):
        Wd = L3.Word(list(self.col), lambda a, b: self.adj(a, b))
        out = []
        for j, F in sorted(Wd.minimal_cyclic()):
            out.append((j + 1, tuple((a + 1, b + 1) for a, b in F)))
        return out

    def patterns(self):
        """which of the patterns T, B, N (03 §1.5) occur, by the 4-cycle configurations with 3 pairs."""
        found = set()
        for j, F in self.minimal_cyclic():
            if len(F) != 3:
                continue
            found.add(self.pattern_of(j, F))
        return found

    def pattern_of(self, j, F):
        cj = self.c(j)
        cols = [self.c(a) for a, b in F]
        if len(set(cols)) < 3:
            return "N"              # two nested pairs of one colour d ~ c(j) around the mover
        (a, b), = [p for p in F if self.c(p[0]) == cj]
        return "T" if b < j else "B"


def cyclic_words(ct, m):
    W0 = WordData(ct, 1, [1])
    n = ct[1]
    out = []
    for w in itertools.product(range(1, n + 1), repeat=m):
        Wd = L3.Word(list(w), lambda a, b: W0.adj(a, b))
        if Wd.search() is not None:
            out.append(w)
    return out


if __name__ == "__main__":
    # self-test against the counts of 04_first_order.md Thm 9.1(a) (1221122, red 2):
    # dot-free homogeneous elements of hom. degree -1 by r-degree: 2447, 1464, 204, 0, 7
    import time
    t0 = time.time()
    W = WordData(["A", 2], 2, [1, 2, 2, 1, 1, 2, 2])
    cands = W.candidates(-1)
    print("1221122 red 2: dot-free homogeneous elements by r-degree:",
          dict(sorted(Counter(rdeg(c[3]) for c in cands).items())), "[%.1fs]" % (time.time() - t0))
    tree = W.all_tree_terms()
    cs = set(cands)
    print("tree terms %d, all of them candidates: %s" % (len(tree), all(k in cs for k in tree)))
    print("hom. degree 0, r-degree >= 1:", dict(sorted(Counter(rdeg(c[3]) for c in W.candidates(0, rmin=1)).items())))
    print("cyclic:", W.is_cyclic(), "minimal cyclic configurations:", W.minimal_cyclic(), "patterns", W.patterns())
