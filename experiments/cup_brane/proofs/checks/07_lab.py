"""
07_lab.py -- symbolic labelled calculus (06_main_intertwiner.md, section 3) over the fraction field
of the polynomial ring in the labelled dots and the parameters.  No klrw needed (sage only for the
rings); the expansion is the one proved in 06 (Props 3.2-3.5, Lemma 2.1) together with
    psi = (x_a - x_b)^{-1} (phi + r)      (two strands of equal colour, a left of b, dots before),
so the result is the unique expansion of an element of e A e' in the Pol[Delta^{-1}]-basis of
brackets D(src -> t) (06 Prop 9.3).

An element with source labelled state ``src`` is a dict  t -> coefficient, t a labelled state
(tuple of labels: slots 1..m and "W"), meaning  sum_t coeff(t) * D(src -> t), coefficients written
at the source in the labelled dots x_s (s = label = slot of the source vertex).

    L = Lab(["A", 2], 1, (1, 1, 1))
    L.std(I)                      labelled state of the vertex I
    L.word_expand(src, word, intertwine=False, start=None)
    L.tree_terms(I)               tree formula of the whole word at I: list of Term
    L.h_terms(I, k)               Conjecture H formula at I (step k): list of Term
    L.Phi_term(I, j), L.X_term(I, j)
    L.expand(I, [term1, term2, ...])   expansion of the product (first term1)
"""
import os, sys, itertools
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from formula import Stack, perm_sign            # noqa: E402
from degree_model import edges as dynkin_edges  # noqa: E402
from sage.all import PolynomialRing, QQ         # noqa: E402


def basis_braid(left, right):
    """klrw basis braid (01_foundations Def. B.10 = klrw.stable_envelopes.braid_from_marked_state):
    left, right are lists of distinct marks; returns the word (1-based positions, first letter first)."""
    perm = {i: right.index(mk) for i, mk in enumerate(left)}
    rw = []
    for i in range(len(left)):
        ell = sum(1 for p in range(i) if perm[p] > perm[i])
        rw += [j + 1 for j in range(i - 1, i - ell - 1, -1)]
    return tuple(rw)


class Term:
    """coefficient (sign * monomial * polynomial in the slot-dots of the source vertex) times the
    basis braid of the slot bijection sigma (or a given word) from src to tgt."""

    def __init__(self, src, tgt, coeff, word, intertwine=False, kind="", info=None):
        self.src, self.tgt, self.coeff, self.word = frozenset(src), frozenset(tgt), coeff, tuple(word)
        self.intertwine, self.kind, self.info = intertwine, kind, info

    def __repr__(self):
        return "%s%s" % (self.kind, self.info if self.info is not None else "")


class Lab:
    def __init__(self, ct, red, word):
        self.ct, self.red, self.word = ct, red, tuple(word)
        self.m = len(self.word)
        self.E = dynkin_edges(ct)
        self.stack = Stack(ct, self.word)
        cols = sorted(set(self.word) | {red})
        self.cols = cols
        names = ["x%d" % s for s in range(1, self.m + 1)]
        names += ["r_%d" % c for c in cols]
        names += ["t_%d_%d" % (c, c2) for c in cols for c2 in cols if frozenset((c, c2)) in self.E]
        names += ["u_%d" % red]
        self.P = PolynomialRing(QQ, names)
        self.F = self.P.fraction_field()
        self.g = dict(zip(names, self.P.gens()))

    # ---- basic data ------------------------------------------------------------------
    def col(self, s):
        return "W" if s == "W" else self.word[s - 1]

    def adj(self, c, c2):
        return frozenset((c, c2)) in self.E

    def x(self, s):
        return self.g["x%d" % s]

    def r(self, c):
        return self.g["r_%d" % c]

    def t(self, c, c2):
        return self.g["t_%d_%d" % (c, c2)]

    def u(self):
        return self.g["u_%d" % self.red]

    def std(self, I):
        I = frozenset(I)
        return tuple([s for s in range(self.m, 0, -1) if s not in I] + ["W"] + [s for s in range(1, self.m + 1) if s in I])

    def q(self, a, b):
        """double crossing polynomial q_{ab} of the labelled strands a, b (labelled dots)."""
        if a == "W":
            a, b = b, a
        if b == "W":
            return self.u() * self.x(a) if self.col(a) == self.red else self.P.one()
        c, c2 = self.col(a), self.col(b)
        if c == c2:
            return self.r(c) ** 2
        if self.adj(c, c2):
            return self.t(c, c2) * self.x(a) + self.t(c2, c) * self.x(b)
        return self.P.one()

    def mono(self, mono):
        el = self.P.one()
        for name, p in mono:
            el *= self.g[name] ** p
        return el

    # ---- the calculus ----------------------------------------------------------------
    def mul_letter(self, src, el, i, intertwine):
        """right multiplication by the crossing of positions i, i+1 (psi, or phi for equal colours if intertwine)."""
        rk = {x: n for n, x in enumerate(src)}
        out = defaultdict(lambda: self.F.zero())
        for t, f in el.items():
            a, b = t[i - 1], t[i]
            new = t[:i - 1] + (b, a) + t[i + 1:]
            inv = rk[a] > rk[b]           # a, b already inverted w.r.t. src: crossing them back gives q
            fac = self.q(a, b) if inv else self.P.one()
            same = a != "W" and b != "W" and self.col(a) == self.col(b)
            if same and not intertwine:
                d = self.x(a) - self.x(b)
                out[new] += f * fac / d
                out[t] += f * self.r(self.col(a)) / d
            else:
                out[new] += f * fac
        return {k: v for k, v in out.items() if v != 0}

    def mul_poly(self, el, poly, canon):
        """right multiplication by a polynomial in the positional dots of the current state, given in the
        slot-variables x_y of the canonical labelled state ``canon`` (x_y = dot at the position of y in canon)."""
        pos = {y: n for n, y in enumerate(canon)}
        poly = self.P(poly)
        out = {}
        for t, f in el.items():
            sub = {self.x(y): self.x(t[pos[y]]) for y in canon if y != "W"}
            out[t] = f * poly.subs(sub)
        return {k: v for k, v in out.items() if v != 0}

    def word_expand(self, src, word, intertwine=False, start=None):
        el = {tuple(src): self.F.one()} if start is None else start
        for i in word:
            el = self.mul_letter(src, el, i, intertwine)
        return el

    def expand(self, I, terms):
        """expansion of the product term_1 * term_2 * ... (first term_1) from std(I)."""
        src = self.std(I)
        el = {src: self.F.one()}
        cur = frozenset(I)
        for T in terms:
            assert T.src == cur, (T, T.src, cur)
            el = self.mul_poly(el, T.coeff, self.std(cur))
            el = self.word_expand(src, T.word, T.intertwine, start=el)
            cur = T.tgt
        return el

    def tau_of(self, I, K, t):
        """labelled state t (labels = slots of I) at the vertex K -> slot bijection tau (tuple, tau[s-1])."""
        sK = self.std(K)
        tau = [None] * self.m
        for p, lab in enumerate(t):
            if lab != "W":
                tau[lab - 1] = sK[p]
        return tuple(tau)

    # ---- terms -------------------------------------------------------------------------
    def braid_word(self, I, J, sigma):
        left = list(self.std(I))
        inv = {sigma[s - 1]: s for s in range(1, self.m + 1)}
        right = [x if x == "W" else inv[x] for x in self.std(J)]
        return basis_braid(left, right)

    def tree_terms(self, I):
        out = []
        for J, sigma, mono, sign in self.stack.terms_from(I):
            (j,) = [s for s in I if s not in J and sigma[s - 1] == s]
            F = tuple(sorted((a, b) for a, b in ((s, sigma[s - 1]) for s in range(1, self.m + 1)) if a < b))
            out.append(Term(I, J, sign * self.mono(mono), self.braid_word(I, J, sigma), kind="T", info=(j, F)))
        return out

    def move_word(self, I, j):
        lab = list(self.std(I))
        target = list(self.std(frozenset(I) - {j}))
        p, q = lab.index(j), target.index(j)
        return tuple(range(p, q, -1))

    def Phi_term(self, I, j):
        I = frozenset(I)
        s = -1 if sum(1 for i in I if i > j) % 2 else 1
        return Term(I, I - {j}, s, self.move_word(I, j), intertwine=True, kind="Phi", info=j)

    def X_term(self, I, j):
        I = frozenset(I)
        s = -1 if sum(1 for i in I if i > j) % 2 else 1
        return Term(I, I - {j}, s, self.move_word(I, j), intertwine=False, kind="X", info=j)

    def h_terms(self, I, k=None, hanging=True):
        """Conjecture H (06 section 7.2) at the vertex I, step k (default m): terms with a chain (s >= 1)."""
        k = self.m if k is None else k
        I = frozenset(I)
        if k not in I:
            return []
        c = self.col(k)
        same = [a for a in range(1, k) if self.col(a) == c]
        free = [a for a in same if a not in I]
        other_pairs = [(a, b) for a in range(1, k) for b in range(a + 1, k)
                       if self.col(a) == self.col(b) and a not in I and b in I]
        out = []
        for s in range(1, len(free) + 1):
            for chain in itertools.combinations(free, s):
                zc = self.P.one()
                for a in same:
                    if a not in chain:
                        zc *= self.x(a) - self.x(k)
                pts = list(chain) + [k]
                arcs = [(pts[i], pts[i + 1]) for i in range(s)]
                used0 = set(pts)
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
                            par = self._hanging_parents(arcs, Gp)
                            if par is None:
                                continue
                            found[Gp] = par
                            new.append(Gp)
                    frontier = new
                for F, par in found.items():
                    sigma = list(range(1, self.m + 1))
                    sigma[k - 1] = chain[0]
                    for (u, v) in arcs:
                        sigma[u - 1] = v
                    img = {k: chain[0]}
                    for a, b in F:
                        sigma[a - 1], sigma[b - 1] = b, a
                        img[b] = a
                    J = frozenset(img.get(x, x) for x in I)
                    eps = perm_sign([img.get(x, x) for x in sorted(I)])
                    coeff = eps * (-self.r(c)) ** s * zc
                    for p, conn in par.items():
                        coeff *= self.r(self.col(p[0])) * self.t(self.col(p[0]), self.col(conn))
                        if conn in I:
                            coeff *= -1
                    out.append(Term(I, J, coeff, self.braid_word(I, J, tuple(sigma)), kind="h", info=(chain, F)))
        return out

    def _hanging_parents(self, arcs, F):
        def inter(p, q):
            (a, b), (a2, b2) = sorted((p, q))
            return a < a2 < b < b2
        links = defaultdict(list)
        n_links = 0
        for p in F:
            for arc in arcs:
                if inter(p, arc):
                    if self.col(p[0]) == self.col(arc[0]):
                        return None
                    if self.adj(self.col(p[0]), self.col(arc[0])):
                        conn = arc[0] if p[0] < arc[0] else arc[1]
                        links["root"].append((p, conn))
                        n_links += 1
        for p, q in itertools.combinations(F, 2):
            if inter(p, q):
                if self.col(p[0]) == self.col(q[0]):
                    return None
                if self.adj(self.col(p[0]), self.col(q[0])):
                    lo, hi = sorted((p, q))
                    links[lo].append((hi, lo[1]))
                    links[hi].append((lo, hi[0]))
                    n_links += 1
        if n_links != len(F):
            return None
        par, st, seen = {}, ["root"], {"root"}
        while st:
            x = st.pop()
            for child, conn in links.get(x, []):
                if child in seen:
                    continue
                seen.add(child); par[child] = conn; st.append(child)
        return par if len(seen) == len(F) + 1 else None

    def zeta(self, k=None):
        k = self.m if k is None else k
        z = self.P.one()
        for a in range(1, k):
            if self.col(a) == self.col(k):
                z *= self.x(a) - self.x(k)
        return z

    # ---- the identity (star) --------------------------------------------------------------
    def star_contributions(self, I):
        """
        All contributions to (star) at the vertex I (m in I), as a list of (K, label, products) where a
        contribution is  sign * expand(products).  The identity is:  sum of all contributions = 0, with
            + Phi_m(I)                       [K = I - m]
            + d_R(I->J) h(J->K)              (d_R = tree terms of w at I with m in J)
            - h(I->J) d_L(J->K)              (d_L = tree terms of w at J, m not in J)
            - zeta_m * F(I->K)               (F = tree terms of w at I with m not in target)
        """
        m = self.m
        I = frozenset(I)
        out = []
        out.append((I - {m}, "Phi", +1, [self.Phi_term(I, m)]))
        Z = Term(I, I, self.zeta(), (), kind="zeta")
        for T in self.tree_terms(I):
            if m in T.tgt:
                for H in self.h_terms(T.tgt):
                    out.append((H.tgt, "dR.h", +1, [T, H]))
            else:
                out.append((T.tgt, "zF", -1, [Z, T]))
        for H in self.h_terms(I):
            for T in self.tree_terms(H.tgt):
                out.append((T.tgt, "h.dL", -1, [H, T]))
        return out

    def star_table(self, I):
        """dict K -> dict tau -> list of (label, products, coefficient)."""
        I = frozenset(I)
        tab = defaultdict(lambda: defaultdict(list))
        for K, label, sgn, prods in self.star_contributions(I):
            el = self.expand(I, prods)
            for t, f in el.items():
                tab[K][self.tau_of(I, K, t)].append((label, prods, sgn * f))
        return tab


def cycles(tau):
    seen, out = set(), []
    for s in range(1, len(tau) + 1):
        if s in seen or tau[s - 1] == s:
            continue
        cyc = [s]; seen.add(s); t = tau[s - 1]
        while t != s:
            cyc.append(t); seen.add(t); t = tau[t - 1]
        out.append("(" + " ".join(map(str, cyc)) + ")")
    return "".join(out) or "id"


def fmt(I):
    return "{" + ",".join(map(str, sorted(I))) + "}"
