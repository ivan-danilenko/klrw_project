"""
Library for proofs/06_main_intertwiner.md.

    import importlib; L = importlib.import_module("06_lib")
    S = L.Setup(["A", 2], 1, (1, 2, 1))

Conventions (README_FOR_PROVERS): a*b = first a, then b; a *labelled state* is a tuple of
labels read from left to right, a label being a slot 1..m or "W" (the red strand).

    S.labels(I)            labelled state of the vertex I
    S.e(lab)               idempotent
    S.psi(lab, i)          (crossing of positions i, i+1 read from lab,  new labelled state)
    S.dot(lab, pos)        dot on the strand at position pos of lab (times the idempotent)
    S.phi(lab, i)          intertwiner (x_i - x_{i+1}) psi_i - r_c   (positions i, i+1 same colour)
    S.X(I, j)              elementary move of slot j (reduced diagram; README Lemma 1)
    S.Phi(I, j)            the same with every same-colour crossing replaced by phi
    S.d(I)                 tree differential out of I:  dict J -> element of e_I A e_J
    S.dPhi(I)              intertwiner differential:    dict J -> element
    S.z(I, x)              prod_{i<x, c(i)=c(x)} (x_x - x_i) * e_I
"""
import os, sys, itertools
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import *            # noqa
from fast import braid_for, exp_for
from formula import Stack, perm_sign
from sample_verify import element as tree_element


def subsets(m, size=None):
    rng = range(1, m + 1)
    sizes = range(m + 1) if size is None else [size]
    for h in sizes:
        for c in itertools.combinations(rng, h):
            yield frozenset(c)


class Setup:
    def __init__(self, ct, red, word, base=ZZ):
        self.ct, self.red, self.word = ct, red, tuple(word)
        self.m = len(self.word)
        P = [(red, self.word)]
        self.KLRW = make_algebra(ct, P, base=base)
        self.D = self.KLRW.base()
        self.cube = Cube(self.KLRW, P, build=False)
        self.stack = Stack(ct, self.word)
        self.States = self.KLRW.KLRWBraid.KLRWstate_set
        self._cache = {}
        self._mono = {}
        self._deg = {}
        self._degPhi = {}

    # ---- states -------------------------------------------------------------
    def col(self, s):
        return self.word[s - 1]

    def labels(self, I):
        I = frozenset(I)
        return tuple([s for s in range(self.m, 0, -1) if s not in I] + ["W"] + [s for s in range(1, self.m + 1) if s in I])

    def node(self, lab):
        return NodeInFramedQuiver(self.red, framing=True) if lab == "W" else NodeInFramedQuiver(self.col(lab))

    def state(self, lab):
        return self.States._element_constructor_(tuple(self.node(x) for x in lab))

    def e(self, lab):
        return self.KLRW.idempotent(self.state(lab))

    def eI(self, I):
        return self.e(self.labels(I))

    # ---- generators ---------------------------------------------------------
    def psi(self, lab, i):
        """crossing of positions i, i+1 (1-based) read from the labelled state lab."""
        el = self.KLRW.element_from_braid((i,), left_state=self.state(lab))
        new = list(lab)
        new[i - 1], new[i] = new[i], new[i - 1]
        return el, tuple(new)

    def dotvar(self, lab, pos):
        st = self.state(lab)
        idx = st.index_among_same_color(pos - 1)
        return self.D.dot_variable(st[pos - 1], idx).monomial

    def dot(self, lab, pos):
        return self.dotvar(lab, pos) * self.e(lab)

    def dot_of(self, lab, slot):
        """dot on the strand labelled ``slot`` in the labelled state."""
        return self.dot(lab, list(lab).index(slot) + 1)

    def r(self, c):
        return self.D.vertex_variable(NodeInFramedQuiver(c)).monomial

    def t(self, c, c2):
        return self.D.edge_variable(NodeInFramedQuiver(c), NodeInFramedQuiver(c2)).monomial

    def u(self, c):
        return self.D.edge_variable(NodeInFramedQuiver(c), NodeInFramedQuiver(c, framing=True)).monomial

    def phi(self, lab, i, variant=0):
        """
        intertwiner on positions i, i+1 of lab (same colour c):
          variant 0:  (x_i - x_{i+1}) psi_i - r_c  ( = -psi_i (x_i - x_{i+1}) + r_c )
        """
        a, b = lab[i - 1], lab[i]
        assert a != "W" and b != "W" and self.col(a) == self.col(b)
        ps, new = self.psi(lab, i)
        el = (self.dot(lab, i) - self.dot(lab, i + 1)) * ps - self.r(self.col(a)) * self.e(lab)
        return el, new

    def word_el(self, lab, word, intertwine=False):
        """product of crossings (1-based positions), first letter first; returns (element, final labels)."""
        el = self.e(lab)
        for i in word:
            a, b = lab[i - 1], lab[i]
            if intertwine and a != "W" and b != "W" and self.col(a) == self.col(b):
                g, lab = self.phi(lab, i)
            else:
                g, lab = self.psi(lab, i)
            el = el * g
        return el, lab

    # ---- elementary moves ---------------------------------------------------
    def move_word(self, I, j):
        I = frozenset(I)
        assert j in I
        lab = list(self.labels(I))
        target = list(self.labels(I - {j}))
        p = lab.index(j)
        q = target.index(j)
        return tuple(range(p, q, -1))      # positions (1-based) p, p-1, ..., q+1

    def X(self, I, j):
        key = ("X", frozenset(I), j)
        if key not in self._cache:
            el, lab = self.word_el(self.labels(I), self.move_word(I, j))
            assert lab == self.labels(frozenset(I) - {j})
            self._cache[key] = el
        return self._cache[key]

    def Phi(self, I, j):
        key = ("Phi", frozenset(I), j)
        if key not in self._cache:
            el, lab = self.word_el(self.labels(I), self.move_word(I, j), intertwine=True)
            assert lab == self.labels(frozenset(I) - {j})
            self._cache[key] = el
        return self._cache[key]

    @staticmethod
    def sign0(I, j):
        return -1 if sum(1 for i in I if i > j) % 2 else 1

    def dPhi(self, I):
        I = frozenset(I)
        return {I - {j}: self.sign0(I, j) * self.Phi(I, j) for j in sorted(I)}

    def d0(self, I):
        I = frozenset(I)
        return {I - {j}: self.sign0(I, j) * self.X(I, j) for j in sorted(I)}

    def d(self, I):
        I = frozenset(I)
        key = ("d", I)
        if key not in self._cache:
            out = defaultdict(lambda: self.KLRW.zero())
            for term in self.stack.terms_from(I):
                out[term[0]] += tree_element(self.cube, self.stack, I, term, self._mono)
            self._cache[key] = dict(out)
        return self._cache[key]

    def term_element(self, I, J, sigma):
        """basis braid of the slot bijection sigma (1-based tuple) from I to J."""
        _, _, braid = braid_for(self.cube, frozenset(I), frozenset(J), tuple(sigma))
        return self.KLRW.monomial(braid)

    # ---- degrees ------------------------------------------------------------
    def deg(self, I):
        """D(I): the shift of P_I in the cube C."""
        I = frozenset(I)
        if I not in self._deg:
            if not I:
                self._deg[I] = self.KLRW.grading_group.zero()
            else:
                j = max(I)
                self._deg[I] = self.deg(I - {j}) + self.X(I, j).degree(check_if_homogeneous=True)
        return self._deg[I]

    def degPhi(self, I):
        I = frozenset(I)
        if I not in self._degPhi:
            if not I:
                self._degPhi[I] = self.KLRW.grading_group.zero()
            else:
                j = max(I)
                self._degPhi[I] = self.degPhi(I - {j}) + self.Phi(I, j).degree(check_if_homogeneous=True)
        return self._degPhi[I]

    def z(self, I, x, lab=None):
        """prod_{i<x, c(i)=c(x)} (x_x - x_i) at the vertex I (or at an arbitrary labelled state)."""
        lab = self.labels(I) if lab is None else lab
        el = self.e(lab)
        for i in range(1, x):
            if self.col(i) == self.col(x):
                el = el * (self.dot_of(lab, x) - self.dot_of(lab, i))
        return el

    def zI(self, I):
        el = self.eI(I)
        for x in sorted(I):
            el = el * self.z(I, x)
        return el

    def zcoI(self, I):
        el = self.eI(I)
        for x in range(1, self.m + 1):
            if x not in I:
                el = el * self.z(I, x)
        return el

    # ---- hom spaces ---------------------------------------------------------
    def _degree_data(self):
        """degree vectors of the variables of the dot algebra and a functional positive on all of them."""
        if not hasattr(self, "_dd"):
            G = self.KLRW.grading_group
            gens = list(self.D.gens())
            vecs = [tuple(int(a) for a in self.D.element_degree(g, G)._vector_()) for g in gens]
            from scipy.optimize import linprog
            import numpy as np
            A = -np.array(vecs, dtype=float)
            res = linprog(np.zeros(len(vecs[0])), A_ub=A, b_ub=-np.ones(len(vecs)), bounds=(-50, 50), method="highs")
            assert res.status == 0
            f = [int(round(4 * a)) for a in res.x]
            w = [sum(a * b for a, b in zip(f, v)) for v in vecs]
            assert all(x > 0 for x in w), w
            self._dd = (gens, vecs, f, w)
        return self._dd

    def monomials_of_degree(self, degree):
        """all monomials in r, t, u and the dots of the given multidegree (own enumeration, no ILP solver)."""
        key = ("mono", degree)
        if key in self._cache:
            return self._cache[key]
        gens, vecs, f, w = self._degree_data()
        target = tuple(int(a) for a in degree._vector_())
        n = len(gens)
        dim = len(target)
        if not hasattr(self, "_last"):
            # last[k] = a coordinate on which no later variable is supported (then the exponent is forced)
            self._last = []
            for k in range(n):
                cand = [i for i in range(dim) if vecs[k][i] and not any(vecs[l][i] for l in range(k + 1, n))]
                self._last.append(cand[0] if cand else None)
        last = self._last
        out = []

        def rec(k, rest, acc):
            if sum(a * b for a, b in zip(f, rest)) < 0:
                return
            if k == n:
                if not any(rest):
                    out.append(tuple(acc))
                return
            i = last[k]
            if i is not None:
                e, rem = divmod(rest[i], vecs[k][i])
                if rem or e < 0:
                    return
                rec(k + 1, tuple(a - e * b for a, b in zip(rest, vecs[k])), acc + [e])
                return
            e = 0
            cur = rest
            while sum(a * b for a, b in zip(f, cur)) >= 0:
                rec(k + 1, cur, acc + [e])
                e += 1
                cur = tuple(a - b for a, b in zip(cur, vecs[k]))

        rec(0, target, [])
        res = []
        for exps in out:
            mon = self.D.one()
            for g, e in zip(gens, exps):
                if e:
                    mon *= g ** e
            res.append(mon)
        self._cache[key] = res
        return res

    def basis(self, I, J, degree, labI=None, labJ=None):
        """basis (dots and parameters) x (basis braids) of the degree-``degree`` part of e_I A e_J."""
        ls = self.state(self.labels(I) if labI is None else labI)
        rs = self.state(self.labels(J) if labJ is None else labJ)
        out = []
        for braid in self.KLRW.KLRWBraid.braids_by_states(ls, rs):
            for mon in self.monomials_of_degree(degree - self.KLRW.braid_degree(braid)):
                out.append(self.KLRW.term(braid, mon))
        return out

    # ---- generic checks -----------------------------------------------------
    def compose(self, A, B):
        """A: dict I -> dict J -> el ; B likewise.  (A*B)_{IK} = sum_J A_{IJ} B_{JK}."""
        out = {}
        for I, row in A.items():
            acc = defaultdict(lambda: self.KLRW.zero())
            for J, a in row.items():
                for K, b in B.get(J, {}).items():
                    acc[K] += a * b
            out[I] = {K: v for K, v in acc.items() if not v.is_zero()}
        return out

    def full(self, f):
        return {I: f(I) for I in subsets(self.m)}


# ---- linear algebra on KLRW elements ---------------------------------------
def coords(el):
    """KLRW element -> dict (left state, right state, word, exponent) -> integer."""
    out = {}
    for braid, coeff in el:
        for exp, c in coeff.iterator_exp_coeff():
            out[(braid.left_state(), braid.right_state(), braid.word(), exp)] = c
    return out


def describe_el(S, el, I=None):
    """human-readable: list of (coeff*monomial, word)."""
    out = []
    for braid, coeff in el:
        out.append("({}) * {}".format(coeff, tuple(braid.word())))
    return " + ".join(out) if out else "0"


# ---- chain maps between two differentials on the cube -------------------------
class ChainMapSystem:
    """
    Unknown G = (G_{IJ}), |J| = |I| + hdeg, G_{IJ} in e_I A e_J homogeneous of degree
    degA(I) - degB(J) + shift.  Equation:  sum_J dA_{IJ} G_{JK}  - sgn * sum_J G_{IJ} dB_{JK} = rhs_{IK}.

    dA, dB: functions I -> dict J -> element;  degA, degB: functions I -> degree.
    ``pairs`` optionally restricts the set of (I, J) carrying unknowns.
    """

    def __init__(self, S, dA, dB, degA, degB, hdeg=0, sgn=1, shift=None, pairs=None, verts=None):
        self.S = S
        self.dA, self.dB, self.sgn, self.hdeg = dA, dB, sgn, hdeg
        m = S.m
        verts = list(subsets(m)) if verts is None else list(verts)
        self.verts = verts
        vset = set(verts)
        shift = S.KLRW.grading_group.zero() if shift is None else shift
        self.vars = []          # (I, J, basis element)
        for I in verts:
            for J in verts:
                if len(J) != len(I) + hdeg:
                    continue
                if pairs is not None and not pairs(I, J):
                    continue
                for b in S.basis(I, J, degA(I) - degB(J) + shift):
                    self.vars.append((I, J, b))
        # incoming dA arrows
        self.into = defaultdict(list)
        for I in verts:
            for J, a in dA(I).items():
                if J in vset:
                    self.into[J].append((I, a))
        self.rows = {}
        self.entries = defaultdict(int)
        for k, (I, J, b) in enumerate(self.vars):
            for I2, a in self.into[I]:
                self._add(k, (I2, J), a * b, 1)
            for K, a in dB(J).items():
                if K in vset:
                    self._add(k, (I, K), b * a, -sgn)

    def _row(self, key):
        if key not in self.rows:
            self.rows[key] = len(self.rows)
        return self.rows[key]

    def _add(self, k, IK, el, s):
        for c, v in coords(el).items():
            self.entries[self._row((IK, c)), k] += s * int(v)

    def matrix(self, extra_rows=0):
        from sage.matrix.constructor import matrix
        from sage.rings.rational_field import QQ
        return matrix(QQ, len(self.rows) + extra_rows, len(self.vars), dict(self.entries), sparse=True)

    def rhs_vector(self, rhs, n_rows=None):
        """rhs: dict (I, K) -> element; returns sage vector (rows that are new are appended)."""
        from sage.modules.free_module_element import vector
        from sage.rings.rational_field import QQ
        vals = {}
        for IK, el in rhs.items():
            for c, v in coords(el).items():
                vals[self._row((IK, c))] = int(v)
        n = len(self.rows)
        vec = vector(QQ, n, sparse=True)
        for i, v in vals.items():
            vec[i] = v
        return vec

    def assemble(self, x):
        """solution vector -> dict I -> dict J -> element."""
        G = defaultdict(lambda: defaultdict(lambda: self.S.KLRW.zero()))
        for k, (I, J, b) in enumerate(self.vars):
            if x[k] != 0:
                G[I][J] += self.S.D(x[k]) * b
        return {I: dict(row) for I, row in G.items()}

    def scipy(self):
        import scipy.sparse as sp
        import numpy as np
        if not self.entries:
            return sp.csr_matrix((len(self.rows), len(self.vars)), dtype=np.int64)
        keys = list(self.entries.keys())
        data = [self.entries[k] for k in keys]
        return sp.csr_matrix((data, ([k[0] for k in keys], [k[1] for k in keys])),
                             shape=(len(self.rows), len(self.vars)), dtype=np.int64)


def fmt_set(I):
    return "{" + ",".join(map(str, sorted(I))) + "}"


def fmt_element(S, I, el):
    """Describe an element of e_I A e_J: for each basis braid the strand bijection and the coefficient."""
    out = []
    lab0 = S.labels(I)
    for braid, coeff in el:
        lab = list(lab0)
        for letter in braid.word():
            lab[letter - 1], lab[letter] = lab[letter], lab[letter - 1]
        out.append("[{}] * ({} => arrive {}; word {})".format(
            coeff, " ".join(map(str, lab0)), " ".join(map(str, lab)), tuple(int(a) for a in braid.word())))
    return "\n      + ".join(out) if out else "0"


# ---- the interpolating cubes C^(k) ------------------------------------------------
def d_mixed(S, k):
    """
    Differential of C^(k): tree formula of the prefix w_{<=k} on the slots <= k, and the
    intertwiner moves Phi_j for the slots j > k (iterated cones of Phi_{k+1}, ..., Phi_m).
    Returns a function I -> dict J -> element.
    """
    def dk(I):
        I = frozenset(I)
        key = ("dmix", k, I)
        if key not in S._cache:
            out = defaultdict(lambda: S.KLRW.zero())
            for term in S.stack.terms_from(I):
                J, sigma, mono, sign = term
                if any(sigma[s - 1] != s for s in range(k + 1, S.m + 1)):
                    continue                      # a pair uses a slot > k
                (j,) = [s for s in I if s not in J and sigma[s - 1] == s]
                if j > k:
                    continue                      # mover > k
                out[J] += tree_element(S.cube, S.stack, I, term, S._mono)
            for j in I:
                if j > k:
                    out[I - {j}] += S.sign0(I, j) * S.Phi(I, j)
            S._cache[key] = dict(out)
        return S._cache[key]
    return dk


def deg_mixed(S, k):
    """grading shift of P_I in C^(k)."""
    def dg(I):
        I = frozenset(I)
        lo = frozenset(s for s in I if s <= k)
        return S.deg(lo) + S.degPhi(I) - S.degPhi(lo)
    return dg


# ---- reading off terms ----------------------------------------------------------
def terms_of(S, I, J, el):
    """
    element of e_I A e_J -> list of (sigma, word, coefficient polynomial); sigma[s-1] = slot of J
    in which the strand starting in slot s of I arrives.
    """
    out = []
    lab0 = S.labels(I)
    labJ = S.labels(J)
    for braid, coeff in el:
        lab = list(lab0)
        for letter in braid.word():
            lab[letter - 1], lab[letter] = lab[letter], lab[letter - 1]
        sigma = [None] * S.m
        for p, s in enumerate(lab):
            if s != "W":
                sigma[s - 1] = labJ[p]
        out.append((tuple(sigma), tuple(int(a) for a in braid.word()), coeff))
    return out


def cycles_str(sigma):
    seen, out = set(), []
    for s in range(1, len(sigma) + 1):
        if s in seen or sigma[s - 1] == s:
            continue
        cyc = [s]
        seen.add(s)
        t = sigma[s - 1]
        while t != s:
            cyc.append(t); seen.add(t); t = sigma[t - 1]
        out.append("(" + " ".join(map(str, cyc)) + ")")
    return "".join(out) or "id"


def print_map(S, G, skip_diagonal=True):
    for I in sorted(G, key=lambda I: (len(I), sorted(I))):
        for J in sorted(G[I], key=sorted):
            if skip_diagonal and I == J:
                continue
            for sigma, word, coeff in terms_of(S, I, J, G[I][J]):
                try:
                    c = coeff.factor()
                except Exception:
                    c = coeff
                print("  {:>12} -> {:<12} sigma {:<16} len {:2d}   {}".format(fmt_set(I), fmt_set(J), cycles_str(sigma), len(word), c))


# ---- closed formula for G when every colour occurs at most twice -----------------
def pairs_of(S):
    """same-colour pairs (a < b) of the word."""
    return [(a, b) for a in range(1, S.m + 1) for b in range(a + 1, S.m + 1) if S.col(a) == S.col(b)]


def adjacent(S, c, c2):
    return frozenset((c, c2)) in S.stack.E


def Qpoly(S, lab, s, s2):
    """bigon polynomial t_{c c'} x_s + t_{c' c} x_{s'} for the strands labelled s, s' of the labelled state lab."""
    c, c2 = S.col(s), S.col(s2)
    return S.t(c, c2) * S.dotvar(lab, list(lab).index(s) + 1) + S.t(c2, c) * S.dotvar(lab, list(lab).index(s2) + 1)


def G_formula_row(S, I):
    """
    Closed formula (06_main_intertwiner.md, Definition of G) for words in which every colour
    occurs at most twice.  Returns dict J -> element of e_I A e_J.
    """
    I = frozenset(I)
    lab = S.labels(I)
    twin = {b: a for a, b in pairs_of(S)}
    avail = [(a, b) for a, b in pairs_of(S) if a not in I and b in I]
    out = defaultdict(lambda: S.KLRW.zero())
    for k in range(len(avail) + 1):
        for F in itertools.combinations(avail, k):
            J = (I - {b for a, b in F}) | {a for a, b in F}
            sigma = list(range(1, S.m + 1))
            img = {b: a for a, b in F}
            for a, b in F:
                sigma[a - 1], sigma[b - 1] = b, a
            eps = perm_sign([img.get(s, s) for s in sorted(I)])
            coeff = S.D(eps)
            for a, b in F:
                coeff *= -S.r(S.col(a))
            for b in I:
                if b in twin and b not in img:
                    coeff *= S.dotvar(lab, list(lab).index(twin[b]) + 1) - S.dotvar(lab, list(lab).index(b) + 1)
            for (a, b), (a2, b2) in itertools.combinations(sorted(F), 2):
                if a < a2 < b < b2 and adjacent(S, S.col(a), S.col(a2)):
                    coeff *= Qpoly(S, lab, b, a2)        # right strand of the earlier pair, left strand of the later
            out[frozenset(J)] += coeff * S.term_element(I, J, sigma)
    return dict(out)


def is_chain_map(S, dA, G, dB, verts=None):
    """sum_J dA_{IJ} G_{JK} == sum_J G_{IJ} dB_{JK} for all I, K.  G: function I -> dict."""
    bad = []
    for I in (subsets(S.m) if verts is None else verts):
        acc = defaultdict(lambda: S.KLRW.zero())
        for J, a in dA(I).items():
            for K, g in G(J).items():
                acc[K] += a * g
        for J, g in G(I).items():
            for K, b in dB(J).items():
                acc[K] -= g * b
        bad += [(I, K) for K, v in acc.items() if not v.is_zero()]
    return bad


# ---- conjectural closed formula for the one-step map (any multiplicity) -----------
def step_formula_row(S, I, k=None, hanging=True):
    """
    Row of the one-step map Gamma^(k) : C^(k-1) -> C^(k) at the vertex I  (k = m by default):
        identity                                     if k not in I,
        z_k + h_k                                    if k in I,
    with the conjectural h_k = sum over chains a_1 < ... < a_s < k (colour of k, not in I) and
    forests of exchange pairs hanging from the chain (see 06_main_intertwiner.md, Conjecture H).
    """
    k = S.m if k is None else k
    I = frozenset(I)
    if k not in I:
        return {I: S.eI(I)}
    lab = S.labels(I)
    c = S.col(k)
    same = [a for a in range(1, k) if S.col(a) == c]
    free = [a for a in same if a not in I]
    xk = S.dotvar(lab, list(lab).index(k) + 1)
    out = defaultdict(lambda: S.KLRW.zero())
    other_pairs = [(a, b) for a, b in pairs_of(S) if b < k and a not in I and b in I]
    for s in range(0, len(free) + 1):
        for chain in itertools.combinations(free, s):
            zc = S.D.one()
            for a in same:
                if a not in chain:
                    zc *= S.dotvar(lab, list(lab).index(a) + 1) - xk
            if s == 0:
                out[I] += zc * S.eI(I)
                continue
            pts = list(chain) + [k]
            arcs = [(pts[i], pts[i + 1]) for i in range(s)]
            used0 = set(pts)
            # grow forests hanging from the chain
            found = {(): {}}
            frontier = [()]
            while frontier and hanging:
                new = []
                for F in frontier:
                    used = used0 | {x for p in F for x in p}
                    for p in other_pairs:
                        if p in F or p[0] in used or p[1] in used:
                            continue
                        Gp = tuple(sorted(F + (p,)))
                        if Gp in found:
                            continue
                        par = _hanging_parents(S, arcs, Gp)
                        if par is None:
                            continue
                        found[Gp] = par
                        new.append(Gp)
                frontier = new
            for F, par in found.items():
                sigma = list(range(1, S.m + 1))
                sigma[k - 1] = chain[0]
                for (u, v) in arcs:
                    sigma[u - 1] = v
                img = {k: chain[0]}
                for a, b in F:
                    sigma[a - 1], sigma[b - 1] = b, a
                    img[b] = a
                J = frozenset(img.get(x, x) for x in I)
                eps = perm_sign([img.get(x, x) for x in sorted(I)])
                coeff = S.D(eps) * (-S.r(c)) ** s * zc
                for p, conn in par.items():
                    coeff *= S.r(S.col(p[0])) * S.t(S.col(p[0]), S.col(conn))
                    if conn in I:
                        coeff *= -1
                out[J] += coeff * S.term_element(I, J, sigma)
    return dict(out)


def _hanging_parents(S, arcs, F):
    """
    link structure of the pairs F hanging from the chain (arcs = consecutive chain arcs, which
    form the root).  Returns dict pair -> connector slot, or None if not a tree / blocked.
    """
    def inter(p, q):
        (a, b), (a2, b2) = sorted((p, q))
        return a < a2 < b < b2
    links = defaultdict(list)
    n_links = 0
    for p in F:
        for arc in arcs:
            if inter(p, arc):
                if S.col(p[0]) == S.col(arc[0]):
                    return None
                if adjacent(S, S.col(p[0]), S.col(arc[0])):
                    conn = arc[0] if p[0] < arc[0] else arc[1]     # slot of the arc inside (a_p, b_p)
                    links["root"].append((p, conn))
                    n_links += 1
    for p, q in itertools.combinations(F, 2):
        if inter(p, q):
            if S.col(p[0]) == S.col(q[0]):
                return None
            if adjacent(S, S.col(p[0]), S.col(q[0])):
                lo, hi = sorted((p, q))
                links[lo].append((hi, lo[1]))      # child hi: connector b_lo lies in (a_hi, b_hi)
                links[hi].append((lo, hi[0]))      # child lo: connector a_hi lies in (a_lo, b_lo)
                n_links += 1
    if n_links != len(F):
        return None
    par, stack, seen = {}, ["root"], {"root"}
    while stack:
        x = stack.pop()
        for child, conn in links.get(x, []):
            if child in seen:
                continue
            seen.add(child); par[child] = conn; stack.append(child)
    return par if len(seen) == len(F) + 1 else None


# ---- a differential stored in the JSON format of cupbrane.run_case ----------------
def d_from_json(S, path):
    """returns a function I -> dict J -> element, for the differential stored at ``path``."""
    import json
    data = json.load(open(path))
    assert tuple(data["strand_colour"]) == S.word
    Braid = S.KLRW.KLRWBraid
    table = defaultdict(lambda: defaultdict(lambda: S.KLRW.zero()))
    for t in data["terms"]:
        I, J = frozenset(t["src"]), frozenset(t["tgt"])
        eps_J = tuple(1 if s + 1 in J else 0 for s in range(S.m))
        eps_I = tuple(1 if s + 1 in I else 0 for s in range(S.m))
        braid = Braid._element_constructor_(S.cube.state(eps_J), tuple(t["word"]))
        assert braid.left_state() == S.cube.state(eps_I)
        coeff = S.D(t["coeff"])
        for name, power in list(t["params"]) + list(t["dots"]):
            coeff *= S.D(name) ** power
        table[I][J] += S.KLRW.term(braid, coeff)
    table = {I: dict(row) for I, row in table.items()}
    return lambda I: table.get(frozenset(I), {})


# ---- the labelled calculus ---------------------------------------------------------
def reduced_word(src, tgt, rng=None):
    """
    A reduced word (1-based positions, first letter first) taking the labelled state src to tgt:
    repeatedly swap an adjacent pair that is in the wrong order.  rng (random.Random) picks the
    pair at random, otherwise the leftmost one is taken.
    """
    rank = {x: i for i, x in enumerate(tgt)}
    cur = list(src)
    word = []
    while True:
        cand = [i for i in range(len(cur) - 1) if rank[cur[i]] > rank[cur[i + 1]]]
        if not cand:
            break
        i = rng.choice(cand) if rng else cand[0]
        cur[i], cur[i + 1] = cur[i + 1], cur[i]
        word.append(i + 1)
    assert tuple(cur) == tuple(tgt)
    return tuple(word)


def Dlab(S, src, tgt, rng=None):
    """the element D(src -> tgt): reduced word, phi for equal colours and psi otherwise."""
    el, lab = S.word_el(tuple(src), reduced_word(src, tgt, rng), intertwine=True)
    assert lab == tuple(tgt)
    return el


def qfactor(S, lab, i, j):
    """double-crossing polynomial q_{ij} of the labelled strands i, j (one may be "W"), dots read at lab."""
    if i == "W":
        i, j = j, i
    xi = S.dotvar(lab, list(lab).index(i) + 1)
    if j == "W":
        return S.u(S.red) * xi if S.col(i) == S.red else S.D.one()
    xj = S.dotvar(lab, list(lab).index(j) + 1)
    c, c2 = S.col(i), S.col(j)
    if c == c2:
        return S.r(c) ** 2
    if adjacent(S, c, c2):
        return S.t(c, c2) * xi + S.t(c2, c) * xj
    return S.D.one()


def inverted_pairs(s, t):
    rk = {x: i for i, x in enumerate(t)}
    return {frozenset((s[i], s[j])) for i in range(len(s)) for j in range(i + 1, len(s)) if rk[s[i]] > rk[s[j]]}


def relabelled_state(S, J, sigma):
    """sigma^{-1} . std(J): the label sigma^{-1}(t) sits where std(J) has the slot t."""
    inv = {sigma[s - 1]: s for s in range(1, S.m + 1)}
    return tuple(x if x == "W" else inv[x] for x in S.labels(J))


def bracket(S, I, J, sigma, rng=None):
    """[sigma]_{I -> J} = D(std I -> sigma^{-1} std J)."""
    return Dlab(S, S.labels(I), relabelled_state(S, J, sigma), rng)
