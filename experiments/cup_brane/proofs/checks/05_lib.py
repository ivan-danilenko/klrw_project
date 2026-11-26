"""
Shared helpers for the checks of proofs/05_main_sliding.md.

    import importlib; L = importlib.import_module("05_lib")
    S = L.Setup(["A", 3], 1, (1, 2, 1, 3))      # cartan type, red colour, word
    S.terms(I)        -> list of Term (mover j, pairs F, parent map, target J, sign, monomial)
    S.el(I, term)     -> the KLRW element  sign * monomial * beta(I; j, F)
    S.beta(I, J, sigma) -> basis braid element (no coefficient)
    S.describe(I, K, element) -> dict  key -> coefficient,   key = (strand map, ) readable

Run with ``sage -python``.
"""
import os, sys, itertools
from collections import defaultdict, namedtuple
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube
from fast import braid_for, exp_for
from formula import Stack, perm_sign

Term = namedtuple("Term", "j F parent J sigma mono sign")


def terms_from(st, I):
    """All tree terms out of the vertex I, with the mover and the pairs made explicit."""
    I = frozenset(I)
    out = []
    for j in sorted(I):
        for F, parent in st.admissible(I, j).items():
            J = (I - {j} - {b for a, b in F}) | {a for a, b in F}
            sigma = list(range(1, st.m + 1))
            for a, b in F:
                sigma[a - 1], sigma[b - 1] = b, a
            out.append(Term(j, F, dict(parent), frozenset(J), tuple(sigma), st.monomial(parent),
                            st.sign(I, j, F, parent)))
    return out


class Setup:
    def __init__(self, ct, red, word):
        self.ct, self.red, self.word = ct, red, tuple(word)
        self.m = len(word)
        P = [(red, self.word)]
        self.K = make_algebra(ct, P)
        self.cube = Cube(self.K, P, build=False)
        self.st = Stack(ct, self.word)
        self.D = self.K.base()
        self._mono = {}

    def terms(self, I):
        return terms_from(self.st, I)

    def mono(self, mono):
        if mono not in self._mono:
            self._mono[mono] = self.D.monomial(*exp_for(self.K, mono)) if mono else self.D.one()
        return self._mono[mono]

    def beta(self, I, J, sigma):
        _, _, braid = braid_for(self.cube, frozenset(I), frozenset(J), sigma)
        return self.K.term(braid, self.D.one())

    def el(self, I, t):
        return self.mono(t.mono) * self.D(t.sign) * self.beta(I, t.J, t.sigma)

    def eps(self, I):
        return tuple(1 if s + 1 in I else 0 for s in range(self.m))

    def state(self, I):
        return self.cube.state(self.eps(I))

    def from_word(self, I, word):
        """Element of the product of crossings ``word`` (1-based letters, first letter first)
        starting at the idempotent of the vertex I."""
        return self.K.element_from_braid(tuple(word), left_state=self.state(I))

    def strand_map(self, I, K, braid):
        """dict: source signed position (+s right of red, -s left) -> target signed position."""
        src = self.cube.labels(self.eps(I))
        tgt = self.cube.labels(self.eps(K))
        lab = list(src)
        for letter in braid.word():
            lab[letter - 1], lab[letter] = lab[letter], lab[letter - 1]
        res = {}
        for pos, l in enumerate(lab):
            if isinstance(l, tuple):
                continue
            s = l + 1
            t = tgt[pos] + 1
            res[s if s in I else -s] = t if t in K else -t
        return res

    def describe(self, I, K, element):
        """dict key -> coefficient (polynomial in parameters and dots); key = sorted tuple of
        the strands that move, as (source signed position, target signed position)."""
        out = {}
        for braid, coeff in element:
            sm = self.strand_map(I, K, braid)
            key = tuple(sorted((s, t) for s, t in sm.items() if s != t))
            assert key not in out
            out[key] = coeff
        return out


def fmt_term(t):
    return "m%d%s" % (t.j, "".join("(%d,%d)" % p for p in t.F))


def fmt_key(key):
    return " ".join("%+d>%+d" % st for st in key)


def words_T1(rank, length):
    """All words of the given length in colours 1..rank with every colour at most twice."""
    for w in itertools.product(range(1, rank + 1), repeat=length):
        if all(w.count(c) <= 2 for c in set(w)):
            yield w


# ---------------------------------------------------------------------------------------------
# strands, crossings, reduced words
# ---------------------------------------------------------------------------------------------
def strand_ids(S, I):
    """Strand ids at the idempotent e_I, left to right: slot numbers (1-based) and 'W'."""
    return ["W" if isinstance(l, tuple) else l + 1 for l in S.cube.labels(S.eps(I))]


def crossings(ids, word):
    """For a word (1-based letters, first letter first) starting at the strands ``ids``:
    list of (strand on the left, strand on the right) just before each letter; final ids."""
    cur = list(ids)
    out = []
    for i in word:
        out.append((cur[i - 1], cur[i]))
        cur[i - 1], cur[i] = cur[i], cur[i - 1]
    return out, cur


def colour(S, s):
    return ("W", S.red) if s == "W" else S.word[s - 1]


def adjacent_colours(S, x, y):
    return frozenset((x, y)) in S.st.E


def check_S1_S2(S, ids, word):
    """Hypotheses of the Straightening Lemma for the word starting at the strands ``ids``:
    (S1) two strands crossing at least twice have different, non-adjacent colours
         (red with a black strand: colour different from the red colour);
    (S2) no three pairwise crossing strands A, B, C with c(A) = c(C) black and
         (B black of adjacent colour, or B red of the same colour).
    Returns (ok1, ok2, number of pairs crossing twice or more)."""
    from collections import Counter
    cr, _ = crossings(ids, word)
    cnt = Counter(frozenset(p) for p in cr)

    def col(s):
        return colour(S, s)

    ok1 = True
    for pair, n in cnt.items():
        if n >= 2:
            x, y = tuple(pair)
            cx, cy = col(x), col(y)
            if "W" in (x, y):
                black = cy if x == "W" else cx
                if black == S.red:
                    ok1 = False
            elif cx == cy or adjacent_colours(S, cx, cy):
                ok1 = False
    ok2 = True
    strands = set(ids)
    for A in strands:
        for C in strands:
            if A == "W" or C == "W" or not (A < C) or col(A) != col(C) or frozenset((A, C)) not in cnt:
                continue
            for B in strands:
                if B in (A, C) or frozenset((A, B)) not in cnt or frozenset((B, C)) not in cnt:
                    continue
                if B == "W":
                    if col(A) == S.red:
                        ok2 = False
                elif adjacent_colours(S, col(A), col(B)):
                    ok2 = False
    return ok1, ok2, sum(1 for n in cnt.values() if n >= 2)


def random_reduced_word(ids, final, rng):
    """A random reduced word taking the arrangement ``ids`` to ``final`` (adjacent transpositions
    of out-of-order neighbours, chosen at random)."""
    rank = {s: i for i, s in enumerate(final)}
    cur = list(ids)
    word = []
    while True:
        desc = [i for i in range(len(cur) - 1) if rank[cur[i]] > rank[cur[i + 1]]]
        if not desc:
            return tuple(word)
        i = rng.choice(desc)
        cur[i], cur[i + 1] = cur[i + 1], cur[i]
        word.append(i + 1)


def word_of(S, element):
    """The word of an element that is a single basis braid (coefficient ignored)."""
    ((braid, coeff),) = list(element)
    return tuple(braid.word())


def dot(S, I, strand):
    """Dot on the given strand (slot number) at the idempotent e_I."""
    ids = strand_ids(S, I)
    st = S.state(I)
    pos = ids.index(strand)
    idx = st.index_among_same_color(pos)
    return S.D.dot_variable(st[pos], idx).monomial * S.K.idempotent(st)
