"""
Shared helpers for the checks of proofs/04_first_order.md.

    import importlib; L = importlib.import_module("04_lib")
    C = L.Ctx(["A", 2], 2, (1, 2, 1))        # type, red colour, word (slots 1..m)

Conventions are those of ../README_FOR_PROVERS.md: a*b = first a, then b; the vertex I is
the idempotent [slots not in I, decreasing] W [slots in I, increasing]; the elementary move
m_x(I) is the basis braid of the identity bijection from e_I to e_{I - x}; the tree formula
(signs included) is taken from ../../formula.py.  Run everything with ``sage -python``.
"""
import os
import sys
import itertools
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
from cupbrane import make_algebra, Cube, braid_from_marked_state  # noqa: E402
from fast import braid_for, exp_for  # noqa: E402
from formula import Stack, perm_sign  # noqa: E402
from degree_model import Model, edges as dynkin_edges  # noqa: E402


def subsets(m):
    for h in range(m + 1):
        for c in itertools.combinations(range(1, m + 1), h):
            yield frozenset(c)


class Ctx:
    def __init__(self, ct, red, word):
        self.ct = list(ct)
        self.red = red
        self.word = tuple(word)
        self.m = len(self.word)
        P = [(red, self.word)]
        self.K = make_algebra(self.ct, P)
        self.cube = Cube(self.K, P, build=False)
        self.st = Stack(self.ct, self.word)
        self.model = Model(self.ct, P)
        self.D = self.K.base()
        self.names = list(self.D.variable_names())
        self.rpos = [i for i, nm in enumerate(self.names) if nm.startswith("r_")]
        self.ident = tuple(range(1, self.m + 1))
        self.E = dynkin_edges(self.ct)
        self._basis = {}
        self._mono = {}

    # ---------------------------------------------------------------- basics
    def col(self, s):
        return self.word[s - 1]

    def adj(self, c, c2):
        return frozenset((c, c2)) in self.E

    def eps(self, I):
        return tuple(1 if s + 1 in I else 0 for s in range(self.m))

    def basis(self, I, J, sigma):
        """Basis element (KLRW basis braid, no dots) of the bijection sigma from e_I to e_J."""
        key = (frozenset(I), frozenset(J), tuple(sigma))
        if key not in self._basis:
            _, _, braid = braid_for(self.cube, frozenset(I), frozenset(J), tuple(sigma))
            self._basis[key] = self.K.monomial(braid)
        return self._basis[key]

    def mono(self, mono):
        """Base-ring monomial from a tuple of (name, power)."""
        if mono not in self._mono:
            self._mono[mono] = self.D.monomial(*exp_for(self.K, mono)) if mono else self.D.one()
        return self._mono[mono]

    def var(self, name):
        return self.D(name)

    def word_element(self, I, word):
        return self.K.element_from_braid(tuple(word), left_state=self.cube.state(self.eps(I)))

    # ------------------------------------------------------------ the cube
    @staticmethod
    def n_above(I, x):
        return sum(1 for z in I if z > x)

    def move(self, I, x):
        I = frozenset(I)
        return self.basis(I, I - {x}, self.ident)

    def delta0(self, I):
        """dict target -> element"""
        I = frozenset(I)
        return {I - {x}: self.K.base()((-1) ** self.n_above(I, x)) * self.move(I, x) for x in I}

    def tree_terms(self, I, npairs=None):
        """Terms of the tree formula out of I with exactly npairs pairs (all if None):
        list of (J, sigma, mono, sign)."""
        out = []
        for J, sigma, mono, sign in self.st.terms_from(frozenset(I)):
            k = sum(p for nm, p in mono if nm.startswith("r_"))
            if npairs is None or k == npairs:
                out.append((J, sigma, mono, sign))
        return out

    def delta_n(self, I, n):
        """dict target -> element: the part of the tree formula with n pairs."""
        res = defaultdict(lambda: self.K.zero())
        for J, sigma, mono, sign in self.tree_terms(I, n):
            res[J] += (self.D(sign) * self.mono(mono)) * self.basis(I, J, sigma)
        return dict(res)

    # ---------------------------------------------------------- truncation
    def rdeg(self, exp):
        return sum(exp[p] for p in self.rpos)

    def r_part(self, el, deg):
        """The part of el (in the basis) whose coefficient monomials have r-degree == deg
        (deg may be a set of allowed degrees)."""
        degs = {deg} if isinstance(deg, int) else set(deg)
        res = self.K.zero()
        for braid, coeff in el:
            d = {e: c for e, c in coeff.dict().items() if self.rdeg(e) in degs}
            if d:
                res += self.K.term(braid, self.D(d))
        return res

    def max_r_part(self, el, maxdeg):
        return self.r_part(el, set(range(maxdeg + 1)))

    def terms(self, el):
        """list of (braid, exp, coeff)"""
        out = []
        for braid, coeff in el:
            for e, c in coeff.dict().items():
                out.append((braid, e, int(c)))
        return out

    def fmt_exp(self, e):
        parts = []
        for pos, pw in enumerate(e):
            if pw:
                parts.append(self.names[pos] + ("^%d" % pw if pw > 1 else ""))
        return "*".join(parts) or "1"

    # ------------------------------------------------------------ bijections
    def bijection(self, I, J, braid):
        """sigma (1-based tuple): the strand from slot s of I ends in slot sigma[s-1] of J."""
        arrive = self.cube.strand_map(self.eps(I), braid)
        labels = self.cube.labels(self.eps(J))
        sigma = {}
        for a, t in zip(arrive, labels):
            if isinstance(a, tuple):
                assert a == t
                continue
            sigma[a + 1] = t + 1
        return tuple(sigma[s] for s in range(1, self.m + 1))

    def dot_index(self, I, slot):
        """(colour, index among the strands of that colour, counted from the left, 1-based)
        of the strand sitting in slot ``slot`` of the state e_I -- i.e. the name x_c_k of a
        dot placed on that strand at the left end of a basis element starting at e_I."""
        labels = self.cube.labels(self.eps(I))
        c = self.col(slot)
        k = 0
        for lab in labels:
            if isinstance(lab, tuple):
                continue
            if self.col(lab + 1) == c:
                k += 1
                if lab + 1 == slot:
                    return c, k
        raise ValueError

    def dot(self, I, slot):
        c, k = self.dot_index(I, slot)
        return self.D("x_%d_%d" % (c, k))

    def t(self, c, c2):
        return self.D("t_%d_%d" % (c, c2))

    def r(self, c):
        return self.D("r_%d" % c)


def compose(C, f, g):
    """f, g : dict source -> dict target -> element (maps on the cube). (fg) = first f then g."""
    res = defaultdict(lambda: defaultdict(lambda: C.K.zero()))
    for I, row in f.items():
        for J, x in row.items():
            if J in g:
                for K, y in g[J].items():
                    res[I][K] += x * y
    return res


def crossing_counts(C, I, word):
    """For a word (1-based letters) read from e_I: Counter over unordered pairs of strand labels
    (slot numbers, or 'W') of the number of times they cross."""
    labels = [("W" if isinstance(x, tuple) else x + 1) for x in C.cube.labels(C.eps(I))]
    cnt = Counter()
    for a in word:
        u, v = labels[a - 1], labels[a]
        cnt[frozenset((u, v))] += 1
        labels[a - 1], labels[a] = v, u
    return cnt, labels


def all_words(rank, maxlen, minlen=1):
    for L in range(minlen, maxlen + 1):
        for w in itertools.product(range(1, rank + 1), repeat=L):
            yield w
