"""
Referee R06: own implementation (independent of 06_lib.py) of the objects of
proofs/06_main_intertwiner.md, built directly in the KLRW algebra of klrw.

    S = Setup(["A", 3], red, word)
    S.arr(I)              labelled state std(I): list of labels (1-based slots, 'W')
    S.g(arr, i)           generator g_i at the labelled state arr (phi for equal black colours)
    S.D(arr, tgt)         product of g's along a reduced word arr -> tgt (labelled states)
    S.Phi(I, j)           Phi_j(I)
    S.dPhi(I)             dict K -> sum_j sign Phi_j(I)
    S.dtree(I)            dict K -> tree differential component (formula.Stack + sample_verify.element)
    S.G(I, variant)       dict J -> G_{I,J} from the TEXT of Definition 6.2 (variants = perturbations)
    S.check_vertex(I)     compare d_Phi G and G d at I

Conventions (README): a*b = first a then b; dots indexed by position on the side where written.
Run with sage -python from proofs/checks.
"""
import os, sys, itertools, random
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube                      # noqa
from fast import braid_for                                   # noqa
from formula import Stack                                    # noqa
from sample_verify import element                            # noqa
from klrw.framed_dynkin import NodeInFramedQuiver            # noqa
from sage.combinat.root_system.cartan_type import CartanType  # noqa


def dynkin_edges(ct):
    C = CartanType(ct).cartan_matrix()
    idx = list(CartanType(ct).index_set())
    E = set()
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if C[a, b] != 0:
                E.add(frozenset((idx[a], idx[b])))
    return E


def perm_sign(seq):
    s = 1
    seq = list(seq)
    for i in range(len(seq)):
        for k in range(i + 1, len(seq)):
            if seq[i] > seq[k]:
                s = -s
    return s


def reduced_word(arr, tgt, rng=None):
    """A reduced word (1-based letters) from labelled state arr to tgt; random if rng given."""
    rank = {lab: p for p, lab in enumerate(tgt)}
    a = list(arr)
    word = []
    while True:
        inv = [i for i in range(len(a) - 1) if rank[a[i]] > rank[a[i + 1]]]
        if not inv:
            break
        i = rng.choice(inv) if rng else inv[0]
        a[i], a[i + 1] = a[i + 1], a[i]
        word.append(i + 1)
    assert a == list(tgt)
    return word


class Setup:
    def __init__(self, ct, red, word):
        self.ct, self.red, self.word = list(ct), red, tuple(word)
        self.m = len(self.word)
        P = [(red, self.word)]
        self.K = make_algebra(ct, P)
        self.Dt = self.K.base()
        self.States = self.K.KLRWBraid.KLRWstate_set
        self.E = dynkin_edges(ct)
        self.cube = Cube(self.K, P, build=False)
        self.stack = Stack(ct, self.word)
        self.cache = {}
        self._phi, self._dt, self._G = {}, {}, {}
        # pairs (at most one per colour when multiplicities <= 2)
        bycol = defaultdict(list)
        for s, c in enumerate(self.word, 1):
            bycol[c].append(s)
        self.bycol = bycol
        self.pairs = [(a, b) for sl in bycol.values() for a, b in itertools.combinations(sl, 2)]
        self.maxmult = max(len(v) for v in bycol.values())
        self._check_conventions()

    # ---- colours / parameters ---------------------------------------------------------------
    def col(self, lab):
        return "W" if lab == "W" else self.word[lab - 1]

    def adj(self, c, c2):
        return frozenset((c, c2)) in self.E

    def r(self, c):
        return self.Dt.vertex_variable(NodeInFramedQuiver(c)).monomial

    def t(self, c, c2):
        return self.Dt.edge_variable(NodeInFramedQuiver(c), NodeInFramedQuiver(c2)).monomial

    # ---- states and elements ----------------------------------------------------------------
    def arr(self, I):
        I = set(I)
        return [s for s in range(self.m, 0, -1) if s not in I] + ["W"] + [s for s in range(1, self.m + 1) if s in I]

    def state(self, arr):
        nodes = tuple(NodeInFramedQuiver(self.red, framing=True) if lab == "W"
                      else NodeInFramedQuiver(self.word[lab - 1]) for lab in arr)
        return self.States._element_constructor_(nodes)

    def idem(self, arr):
        return self.K.idempotent(self.state(arr))

    def psi(self, arr, i):
        return self.K.element_from_braid((i,), left_state=self.state(arr))

    def dotpos(self, arr, p):
        """dot on the strand at 1-based position p of arr (times idempotent)."""
        st = self.state(arr)
        idx = st.index_among_same_color(p - 1)
        return self.Dt.dot_variable(st[p - 1], idx).monomial * self.idem(arr)

    def dot(self, arr, lab):
        return self.dotpos(arr, arr.index(lab) + 1)

    def g(self, arr, i):
        a, b = arr[i - 1], arr[i]
        if a != "W" and b != "W" and self.col(a) == self.col(b):
            return (self.dotpos(arr, i) - self.dotpos(arr, i + 1)) * self.psi(arr, i) \
                - self.r(self.col(a)) * self.idem(arr)
        return self.psi(arr, i)

    def D(self, arr, tgt, rng=None, word=None):
        arr = list(arr)
        if word is None:
            word = reduced_word(arr, tgt, rng)
        el = self.idem(arr)
        cur = list(arr)
        for i in word:
            el = el * self.g(cur, i)
            cur[i - 1], cur[i] = cur[i], cur[i - 1]
        return el

    def _check_conventions(self):
        # t_{cc'} convention: psi^2 on (c, c') = t_{cc'} x_c + t_{c'c} x_{c'}
        for (c, c2) in [(c, c2) for c in self.bycol for c2 in self.bycol if self.adj(c, c2)][:1]:
            pass  # checked in R06_basic.py on a 2-strand algebra

    # ---- the cube C_Phi -----------------------------------------------------------------------
    def Phi(self, I, j):
        key = (frozenset(I), j)
        if key not in self._phi:
            self._phi[key] = self.D(self.arr(I), self.arr(set(I) - {j}))
        return self._phi[key]

    def dPhi(self, I):
        I = frozenset(I)
        out = {}
        for j in sorted(I):
            sgn = (-1) ** sum(1 for i in I if i > j)
            out[I - {j}] = sgn * self.Phi(I, j)
        return out

    # ---- tree differential ---------------------------------------------------------------------
    def dtree(self, I):
        I = frozenset(I)
        if I not in self._dt:
            out = defaultdict(lambda: self.K.zero())
            for term in self.stack.terms_from(I):
                out[term[0]] += element(self.cube, self.stack, I, term, self.cache)
            self._dt[I] = dict(out)
        return self._dt[I]

    # ---- G of Definition 6.2, written from the text -----------------------------------------
    def interlaced_lt(self, p, q):
        """p < q interlaced: a_p < a_q < b_p < b_q."""
        return p[0] < q[0] < p[1] < q[1]

    def linked(self, p, q):
        return (self.interlaced_lt(p, q) or self.interlaced_lt(q, p)) and self.adj(self.col(p[0]), self.col(q[0]))

    def bracket(self, I, sigma, J):
        """[sigma]_{I->J} = D(std I -> sigma^{-1} std J); sigma: dict slot->slot (missing = fixed)."""
        inv = {v: k for k, v in sigma.items()}
        tgt = [lab if lab == "W" else inv.get(lab, lab) for lab in self.arr(J)]
        return self.D(self.arr(I), tgt)

    def Q(self, c, c2, xa, xb):
        """Q_{c c2}(xa, xb) = t_{c c2} xa + t_{c2 c} xb (xa on colour c) if adjacent, else 1."""
        if self.adj(c, c2):
            return self.t(c, c2) * xa + self.t(c2, c) * xb
        return None

    def G(self, I, variant="text"):
        """
        Row I of G (Definition 6.2).  variant:
          'text'   : as written
          'noQ'    : drop the Q factors
          'LL'     : Q(x_{a_p}, x_{a_q}) instead of Q(x_{b_p}, x_{a_q})
          'RR'     : Q(x_{b_p}, x_{b_q})
          'plusr'  : +r_p instead of -r_p
          'nosign' : drop epsilon(I,F)
          'tree'   : sum only over F connected... (not used)
        """
        I = frozenset(I)
        key = (I, variant)
        if key in self._G:
            return self._G[key]
        a0 = self.arr(I)
        e = self.idem(a0)
        xs = {s: self.dot(a0, s) for s in range(1, self.m + 1)}
        PI = [p for p in self.pairs if p[0] not in I and p[1] in I]
        out = defaultdict(lambda: self.K.zero())
        for k in range(len(PI) + 1):
            for F in itertools.combinations(PI, k):
                used = [x for p in F for x in p]
                if len(set(used)) != len(used):
                    continue
                IF = (I - {p[1] for p in F}) | {p[0] for p in F}
                rep = {p[1]: p[0] for p in F}
                eps = perm_sign([rep.get(s, s) for s in sorted(I)])
                if variant == "nosign":
                    eps = 1
                coef = eps * e
                for p in F:
                    rr = self.r(self.col(p[0]))
                    coef = coef * ((rr if variant == "plusr" else -rr) * e)
                for p in self.pairs:
                    if p not in F and p[1] in I:
                        coef = coef * (xs[p[0]] - xs[p[1]])
                if variant != "noQ":
                    for p in F:
                        for q in F:
                            if self.interlaced_lt(p, q) and self.adj(self.col(p[0]), self.col(q[0])):
                                cp, cq = self.col(p[0]), self.col(q[0])
                                if variant == "LL":
                                    qq = self.Q(cp, cq, xs[p[0]], xs[q[0]])
                                elif variant == "RR":
                                    qq = self.Q(cp, cq, xs[p[1]], xs[q[1]])
                                else:
                                    qq = self.Q(cp, cq, xs[p[1]], xs[q[0]])
                                coef = coef * qq
                sigma = {}
                for p in F:
                    sigma[p[0]], sigma[p[1]] = p[1], p[0]
                out[frozenset(IF)] += coef * self.bracket(I, sigma, IF)
        self._G[key] = dict(out)
        return self._G[key]

    # ---- the identity d_Phi G = G d -----------------------------------------------------------
    def check_vertex(self, I, variant="text"):
        I = frozenset(I)
        lhs = defaultdict(lambda: self.K.zero())
        for J, x in self.dPhi(I).items():
            for K, y in self.G(J, variant).items():
                lhs[K] += x * y
        rhs = defaultdict(lambda: self.K.zero())
        for J, x in self.G(I, variant).items():
            for K, y in self.dtree(J).items():
                rhs[K] += x * y
        bad = [K for K in set(lhs) | set(rhs) if not (lhs[K] - rhs[K]).is_zero()]
        return bad, len(set(lhs) | set(rhs))

    def vertices(self):
        for h in range(1, self.m + 1):
            for I in itertools.combinations(range(1, self.m + 1), h):
                yield frozenset(I)
