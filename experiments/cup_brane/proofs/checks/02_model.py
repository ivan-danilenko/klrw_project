"""
Pure-Python model of the omega_1 I-brane of sl_{n+1} (GLUING.md section 2), used by the
proofs in proofs/02_omega1_ibrane.md.

    python3 proofs/checks/02_model.py N_MAX        # self-test: compare with glue/omega1_*.py

Conventions (see 02_omega1_ibrane.md, section 1):
  * colours 1..n; labels 0 (= red strand W_1) and n+1 (= red strand W_n) are "frozen in B";
  * a vertex is v = (k, A, C) with A <= {1..k}, C <= {k+1..n}; region(c) in {"A","B","C"};
    the four states are A, BL (c <= k, in B), BR (c > k, in B), C;
  * position of a label in the idempotent: A: -c, B: c, C: 2n+2-c, W_1: 0, W_n: n+1
    (the idempotent orders the labels by position);
  * order bit of the edge x (0 <= x <= n): o_x(v) = 1 iff label x is left of label x+1.

Everything here is pure Python (no sage).  Parameters are Laurent monomials, stored as
frozensets of (name, exponent); polynomials in the dots x_1..x_n with Laurent-monomial
coefficients are dicts  (param monomial, dot exponent tuple) -> integer.
"""
import itertools
import os
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------- vertices
def vertices(n):
    for k in range(n + 1):
        for a in range(k + 1):
            for A in itertools.combinations(range(1, k + 1), a):
                for c in range(n - k + 1):
                    for C in itertools.combinations(range(k + 1, n + 1), c):
                        yield (k, frozenset(A), frozenset(C))


def region(v, c):
    k, A, C = v
    if c in A:
        return "A"
    if c in C:
        return "C"
    return "B"


def state(v, c):
    """A, BL, BR or C."""
    r = region(v, c)
    if r != "B":
        return r
    return "BL" if c <= v[0] else "BR"


def states(n, v):
    return tuple(state(v, c) for c in range(1, n + 1))


def from_states(n, st):
    """Inverse of states(): the vertex with the given state word (None if not monotone)."""
    L = [c for c in range(1, n + 1) if st[c - 1] in ("A", "BL")]
    k = len(L)
    if L != list(range(1, k + 1)):
        return None
    A = frozenset(c for c in range(1, n + 1) if st[c - 1] == "A")
    C = frozenset(c for c in range(1, n + 1) if st[c - 1] == "C")
    return (k, A, C)


HDEG = {"A": 1, "BL": 2, "BR": 0, "C": 1}


def hdeg(n, v):
    """Homological degree k + #B_L + #C (GLUING.md) = sum of per-colour degrees."""
    k, A, C = v
    h = k + sum(1 for c in range(1, k + 1) if c not in A) + len(C)
    assert h == sum(HDEG[s] for s in states(n, v))
    return h


def position(n, v, lab):
    if lab == 0:
        return 0
    if lab == n + 1:
        return n + 1
    r = region(v, lab)
    return -lab if r == "A" else lab if r == "B" else 2 * n + 2 - lab


def idempotent(n, v):
    """The colour sequence, left to right; 'W1' / 'Wn' for the red strands."""
    labs = sorted(range(0, n + 2), key=lambda x: position(n, v, x))
    return tuple("W1" if x == 0 else "Wn" if x == n + 1 else x for x in labs)


def left_of(n, v, x, y):
    return position(n, v, x) < position(n, v, y)


def order_bits(n, v):
    return tuple(int(left_of(n, v, x, x + 1)) for x in range(n + 1))


def inversions(n, v, w):
    """Unordered pairs of labels whose relative order differs between v and w."""
    out = set()
    for x, y in itertools.combinations(range(n + 2), 2):
        if left_of(n, v, x, y) != left_of(n, w, x, y):
            out.add((x, y))
    return out


def double_pairs(n, v, u, w):
    """Pairs of labels crossing twice in beta(v,u) beta(u,w)."""
    return inversions(n, v, u) & inversions(n, u, w)


# ----------------------------------------------------------------------------- parameters
def tname(n, a, b):
    """t_{a,b} with t_{1,0} := u_1 and t_{n,n+1} := u_n."""
    if a == 1 and b == 0:
        return "u_1"
    if a == n and b == n + 1:
        return "u_%d" % n
    assert 1 <= a <= n and 1 <= b <= n and abs(a - b) == 1, (a, b)
    return "t_%d_%d" % (a, b)


def mono(d):
    return frozenset((k, v) for k, v in d.items() if v)


def mono_mul(m1, m2):
    c = Counter(dict(m1))
    c.update(dict(m2))
    return mono(c)


def tau_power(n, lo, hi, e):
    """prod_{x=lo..hi} tau_x^e, tau_x = t_{x,x-1}/t_{x,x+1}."""
    c = Counter()
    for x in range(lo, hi + 1):
        c[tname(n, x, x - 1)] += e
        c[tname(n, x, x + 1)] -= e
    return mono(c)


ONE = frozenset()


# ----------------------------------------------------------------------------- exterior algebra signs
def gen_order(n):
    """a_1 < ... < a_n < g_n < ... < g_1."""
    return lambda y: y[1] if y[0] == "a" else 2 * n + 2 - y[1]


def ext_monomial(n, v):
    gens = set()
    for c in range(1, n + 1):
        s = state(v, c)
        if s == "A":
            gens.add(("a", c))
        elif s == "C":
            gens.add(("g", c))
        elif s == "BL":
            gens |= {("a", c), ("g", c)}
    return gens


def apply_ops(n, v, ops):
    """Apply operators (rightmost FIRST in the written product = first in the list here)."""
    order = gen_order(n)
    gens = ext_monomial(n, v)
    s = 1
    for op, c in ops:
        if op in ("ia", "ig"):
            x = ("a" if op == "ia" else "g", c)
            assert x in gens, (v, ops)
            s *= (-1) ** sum(1 for y in gens if order(y) < order(x))
            gens = gens - {x}
        elif op == "eg":
            x = ("g", c)
            assert x not in gens
            s *= (-1) ** sum(1 for y in gens if order(y) < order(x))
            gens = gens | {x}
        elif op == "rho":
            ops2 = [("ia", c), ("eg", c)]
            for op2, c2 in ops2:
                x = ("a" if op2 == "ia" else "g", c2)
                if op2 == "ia":
                    assert x in gens
                    s *= (-1) ** sum(1 for y in gens if order(y) < order(x))
                    gens = gens - {x}
                else:
                    assert x not in gens
                    s *= (-1) ** sum(1 for y in gens if order(y) < order(x))
                    gens = gens | {x}
        else:
            raise ValueError(op)
    return gens, s


# ----------------------------------------------------------------------------- the terms
class Term:
    __slots__ = ("src", "tgt", "typ", "d", "extra", "sign", "coeff")

    def __init__(self, src, tgt, typ, d, extra, sign, coeff):
        self.src, self.tgt, self.typ, self.d, self.extra, self.sign, self.coeff = src, tgt, typ, d, extra, sign, coeff

    def tag(self):
        return (self.typ, self.d, self.extra)

    def __repr__(self):
        return "Term(%s, d=%d, extra=%s, %+d, %s)" % (self.typ, self.d, self.extra, self.sign, dict(self.coeff))


def terms_from(n, v):
    """All terms out of v, written from the text of GLUING.md section 2 (independently of glue/)."""
    k, A, C = v
    st = states(n, v)
    out = []
    # internal
    for c in range(1, n + 1):
        if st[c - 1] == "BL":
            w = (k, A | {c}, C)
            out.append(Term(v, w, "int", 0, c, apply_ops(n, v, [("ig", c)])[1], ONE))
        elif st[c - 1] == "C":
            w = (k, A, C - {c})
            out.append(Term(v, w, "int", 0, c, apply_ops(n, v, [("ig", c)])[1], ONE))
    # gluing
    for d in range(1, k + 1):
        l = k - d
        sw = list(range(l + 1, k + 1))
        if all(st[c - 1] == "A" for c in sw):                          # (i)
            w = (l, A - set(sw), C | set(sw[:-1]))
            ops = [("rho", c) for c in sw[:-1]] + [("ia", k)]
            gens, s = apply_ops(n, v, ops)
            assert gens == ext_monomial(n, w)
            out.append(Term(v, w, "i", d, None, (-1) ** (d + 1) * s, ONE))
        if st[l] == "BL" and all(st[c - 1] == "A" for c in sw[1:]):    # (ii) and (iii)
            w = (l, A - set(sw), C | set(sw))
            ops = [("ia", l + 1)] + [("rho", c) for c in sw[1:]]
            gens, s = apply_ops(n, v, ops)
            assert gens == ext_monomial(n, w)
            out.append(Term(v, w, "ii", d, None, s, tau_power(n, l + 1, k, 1)))
            A2, C2 = (A, C) if d == 1 else (A - set(sw), C | set(sw[:-1]))
            for c in range(1, n + 1):
                if c <= l and st[c - 1] == "A":
                    w = (l, A2 - {c}, C2)
                    sgn0 = (-1) ** ((l + 1 - c) + d)
                    cf = tau_power(n, c, l, -1)
                elif c > k and st[c - 1] == "BR":
                    w = (l, A2, C2 | {c})
                    sgn0 = (-1) ** (c - k)
                    cf = tau_power(n, l + 1, c, 1)
                else:
                    continue
                if d == 1:
                    ops = [("ia", k), ("ig", k), ("eg", c)]
                else:
                    ops = [("ia", l + 1)] + [("rho", x) for x in sw[1:-1]] + [("ia", k), ("eg", c)]
                gens, s = apply_ops(n, v, ops)
                assert gens == ext_monomial(n, w)
                out.append(Term(v, w, "iii", d, c, sgn0 * s, cf))
    for t in out:
        assert hdeg(n, t.tgt) == hdeg(n, v) - 1
    return out


def all_terms(n):
    return [t for v in vertices(n) for t in terms_from(n, v)]


# ----------------------------------------------------------------------------- polynomials
def Q(n, x):
    """Bigon polynomial of the edge x (labels x, x+1): t_{x,x+1} x_x + t_{x+1,x} x_{x+1}, x_0 = x_{n+1} = 0."""
    p = {}
    if 1 <= x:
        e = [0] * n; e[x - 1] = 1
        p[(mono({tname(n, x, x + 1): 1}), tuple(e))] = 1
    if x + 1 <= n:
        e = [0] * n; e[x] = 1
        key = (mono({tname(n, x + 1, x): 1}), tuple(e))
        p[key] = p.get(key, 0) + 1
    return p


def poly_mul(p, q):
    out = defaultdict(int)
    for (m1, e1), c1 in p.items():
        for (m2, e2), c2 in q.items():
            out[(mono_mul(m1, m2), tuple(a + b for a, b in zip(e1, e2)))] += c1 * c2
    return {k: v for k, v in out.items() if v}


def poly_add(p, q, s=1):
    out = defaultdict(int, p)
    for k, v in q.items():
        out[k] += s * v
    return {k: v for k, v in out.items() if v}


def poly_const(n, m, c=1):
    return {(m, (0,) * n): c}


def bigon_factor(n, pairs):
    """Product of the bigon polynomials of a set of label pairs (only adjacent labels matter)."""
    p = poly_const(n, ONE)
    for (x, y) in pairs:
        if y == x + 1 and not (x == 0 and y == n + 1):
            p = poly_mul(p, Q(n, x))
        # any other pair: psi^2 = 1 (non-adjacent colours, or a black strand and a red
        # strand of a different colour); the pair (W_1, W_n) never crosses.
    return p


# ----------------------------------------------------------------------------- d^2
def d_squared(n, terms=None):
    """Returns dict (v, w) -> list of (term1, term2, double pairs) and the dict of sums."""
    if terms is None:
        terms = all_terms(n)
    out_of = defaultdict(list)
    for t in terms:
        out_of[t.src].append(t)
    paths = defaultdict(list)
    for t1 in terms:
        for t2 in out_of[t1.tgt]:
            paths[(t1.src, t2.tgt)].append((t1, t2, double_pairs(n, t1.src, t1.tgt, t2.tgt)))
    sums = {}
    for key, lst in paths.items():
        s = {}
        for t1, t2, D in lst:
            p = bigon_factor(n, D)
            m = mono_mul(t1.coeff, t2.coeff)
            p = poly_mul(p, poly_const(n, m, t1.sign * t2.sign))
            s = poly_add(s, p)
        sums[key] = s
    return paths, sums


# ----------------------------------------------------------------------------- self-test
def compare_with_glue(n):
    """Compare targets, signs and coefficients with glue/omega1_rule.py, glue/omega1_formula.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "glue"))
    import omega1_rule, omega1_formula
    mine = {}
    for t in all_terms(n):
        key = (t.src, t.tgt)
        assert key not in mine
        mine[key] = (t.sign, dict(t.coeff))
    theirs = {}
    for v in omega1_rule.vertices(n):
        for w, tag in omega1_rule.targets(n, v):
            theirs[(v, w)] = (omega1_formula.rule_sign(n, v, tag), omega1_formula.coefficient(n, v, tag))
    assert set(mine) == set(theirs), (n, len(mine), len(theirs))
    bad = [key for key in mine if mine[key][0] != theirs[key][0] or mine[key][1] != theirs[key][1]]
    return len(mine), len(bad)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for n in range(1, N + 1):
        nt, nbad = compare_with_glue(n)
        paths, sums = d_squared(n)
        nonzero = sum(1 for s in sums.values() if s)
        print("n=%d (sl_%d): %d vertices, %d terms, agree with glue/: %s;  d^2 components %d, non-zero %d"
              % (n, n + 1, sum(1 for _ in vertices(n)), nt, nbad == 0, len(sums), nonzero), flush=True)
