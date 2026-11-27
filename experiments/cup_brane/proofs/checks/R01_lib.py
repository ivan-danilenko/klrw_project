"""
R01_lib.py -- referee's own helpers (does not import any 01_*.py).

* Tokens: black strand of colour c = int c; red strand = ("W", k).
* PolyRepR: the polynomial representation of 01 Def. A.2, written from the formulas, acting on
  polynomials over a ring B (GF(p) after specialising the parameters, or ZZ[params]).
* klrw conversion helpers (through proofs/checks/alg.py, the shared interface of the README).
"""
import itertools
import random
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def is_red(a):
    return isinstance(a, tuple)


def tok_str(a):
    return "W%d" % a[1] if is_red(a) else str(a)


def seq_str(s):
    return " ".join(tok_str(a) for a in s)


def parse_seq(txt):
    out = []
    for t in txt.split():
        out.append(("W", int(t[1:])) if t[0] == "W" else int(t))
    return tuple(out)


def node_tok(node):
    return ("W", node.node) if node.is_framing() else node.node


# ---------------------------------------------------------------------------------------
# Dynkin data
# ---------------------------------------------------------------------------------------
def dynkin_edges(ct):
    typ, n = ct
    E = set()
    if typ == "A":
        E = {frozenset((i, i + 1)) for i in range(1, n)}
    elif typ == "D":
        E = {frozenset((i, i + 1)) for i in range(1, n - 1)} | {frozenset((n - 2, n))}
    elif typ == "E":
        # Bourbaki: 1-3-4-5-6(-7-8), 2-4
        E = {frozenset((1, 3)), frozenset((3, 4)), frozenset((4, 5)), frozenset((2, 4))}
        for i in range(5, n):
            E.add(frozenset((i, i + 1)))
    return E


# ---------------------------------------------------------------------------------------
# polynomial representation (own implementation)
# ---------------------------------------------------------------------------------------
class PolyRepR:
    """
    Right action on V_s = B[y_0..y_{n-1}] (only black positions carry variables that occur).
    params: dict with keys ('r',c), ('t',a,b), ('u',c) -> elements of B.
    """

    def __init__(self, ct, n, params, B, rank=None):
        from sage.all import PolynomialRing
        self.E = dynkin_edges(ct)
        self.n = n
        self.B = B
        self.R = PolynomialRing(B, ["y%d" % i for i in range(n)])
        self.y = list(self.R.gens())
        self.p = params
        self.rank = rank if rank is not None else (lambda a: 1000 if is_red(a) else a)

    def adj(self, a, b):
        return frozenset((a, b)) in self.E

    def Q(self, a, b, yi, yj):
        if is_red(a):
            return self.p[("u", b)] * yj if b == a[1] else self.R(1)
        if is_red(b):
            return self.p[("u", a)] * yi if a == b[1] else self.R(1)
        if a == b:
            return self.R(0)
        if self.adj(a, b):
            return self.p[("t", a, b)] * yi + self.p[("t", b, a)] * yj
        return self.R(1)

    def swap(self, f, i):
        if not hasattr(self, "_sw"):
            self._sw = {}
        if i not in self._sw:
            g = list(self.y)
            g[i], g[i + 1] = g[i + 1], g[i]
            self._sw[i] = self.R.hom(g, self.R)
        return self._sw[i](f)

    def psi(self, s, i, f):
        """f . psi_{i,s}, i 0-based (positions i, i+1). Returns (target, poly)."""
        a, b = s[i], s[i + 1]
        t = list(s)
        t[i], t[i + 1] = b, a
        t = tuple(t)
        if a == b and not is_red(a):
            num = f - self.swap(f, i)
            q, rem = num.quo_rem(self.y[i] - self.y[i + 1])
            assert rem == 0
            return t, self.p[("r", a)] * q
        if self.rank(a) < self.rank(b):
            return t, self.swap(f, i)
        return t, self.Q(t[i], t[i + 1], self.y[i], self.y[i + 1]) * self.swap(f, i)

    def word(self, s, w, f):
        """w: letters 0-based."""
        for i in w:
            s, f = self.psi(s, i, f)
        return s, f


def random_params(ct, rng, B, nodes, zero=()):
    """random non-zero values in B for all parameters; names in `zero` are set to 0."""
    E = dynkin_edges(ct)
    p = {}
    for c in nodes:
        p[("r", c)] = B(rng.randint(1, 10 ** 6))
        p[("u", c)] = B(rng.randint(1, 10 ** 6))
        for d in nodes:
            if frozenset((c, d)) in E:
                p[("t", c, d)] = B(rng.randint(1, 10 ** 6))
    for key in zero:
        p[key] = B(0)
    return p


# ---------------------------------------------------------------------------------------
# klrw side
# ---------------------------------------------------------------------------------------
def klrw_coeff_to_poly(A, rep, s, coeff):
    """klrw coefficient (dots of the left state s, by colour index; parameters) -> rep.R."""
    names = A.D.variable_names()
    res = rep.R(0)
    for exp, c in coeff.iterator_exp_coeff():
        term = rep.R(rep.B(c))
        for pos, power in enumerate(exp):
            if power == 0:
                continue
            nm = names[pos]
            parts = nm.split("_")
            if parts[0] == "x":
                col, idx = int(parts[1]), int(parts[2])
                q = [i for i in range(len(s)) if (not is_red(s[i])) and s[i] == col][idx - 1]
                term *= rep.y[q] ** power
            elif parts[0] == "r":
                term *= rep.p[("r", int(parts[1]))] ** power
            elif parts[0] == "t":
                term *= rep.p[("t", int(parts[1]), int(parts[2]))] ** power
            elif parts[0] == "u":
                term *= rep.p[("u", int(parts[1]))] ** power
            else:
                raise ValueError(nm)
        res += term
    return res


def act_klrw(A, rep, s, element, f):
    """rho(element)(f), f in V_s, element of e(s) A; result dict target state -> poly."""
    out = {}
    for braid, coeff in element:
        left = tuple(node_tok(x) for x in braid.left_state())
        assert left == s, (left, s)
        g = klrw_coeff_to_poly(A, rep, s, coeff) * f
        t, h = rep.word(s, tuple(i - 1 for i in braid.word()), g)
        out[t] = out.get(t, rep.R(0)) + h
    return {t: h for t, h in out.items() if h != 0}


def compose_act(A, rep, s, a, b, f):
    mid = act_klrw(A, rep, s, a, f)
    out = {}
    for t, g in mid.items():
        for t2, h in act_klrw(A, rep, t, b, g).items():
            out[t2] = out.get(t2, rep.R(0)) + h
    return {t: h for t, h in out.items() if h != 0}


def lehmer_word(perm):
    """01 Def. B.10 written independently: perm[p] = target position (0-based) of the strand
    at source position p.  Returns 0-based letters."""
    n = len(perm)
    w = []
    for p in range(n):
        l = sum(1 for q in range(p) if perm[q] > perm[p])
        w += [p - 1 - k for k in range(l)]
    return w


def random_reduced_word(perm, rng):
    """a random reduced word (0-based letters, applied left to right from the source) whose
    product moves the strand at source position p to perm[p]."""
    n = len(perm)
    cur = list(range(n))       # cur[i] = source label of the strand now at position i
    word = []
    while True:
        cand = [i for i in range(n - 1) if perm[cur[i]] > perm[cur[i + 1]]]
        if not cand:
            break
        i = rng.choice(cand)
        cur[i], cur[i + 1] = cur[i + 1], cur[i]
        word.append(i)
    return word


def colour_perms(s, t):
    """all permutations perm (perm[p] = target position) with t[perm[p]] = s[p]."""
    n = len(s)
    if sorted(map(str, s)) != sorted(map(str, t)):
        return []
    out = []
    for perm in itertools.permutations(range(n)):
        if all(t[perm[p]] == s[p] for p in range(n)):
            out.append(perm)
    return out


def states_of(seq):
    return sorted(set(itertools.permutations(seq)), key=str)
