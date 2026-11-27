"""
01_polyrep.py   (sage -python proofs/checks/01_polyrep.py [relations|hom|rank|all])

The polynomial representation of Definition A.6 of 01_foundations.md, implemented from
the formulas alone (independently of klrw), and three checks:

  relations : the operators satisfy every defining relation of Definition A.1 (all colour
              patterns of 2 and 3 strands over {1,2,3} of A_3 and one red strand W1 or W2),
              tested on all monomials of degree <= 3, over the polynomial parameter ring.
  hom       : the klrw implementation is a representation-compatible algebra: for random
              basis elements a, b of the implementation,  rho(a*b) = rho(b) o rho(a)
              (right action: first a, then b) on test polynomials.
  rank      : linear independence part of the basis theorem (F1): for several pairs of
              states the elements  x^alpha * beta_w  (w over all basis braids, |alpha| <= D)
              act by linearly independent operators (rank over GF(p) after a random
              specialisation of the parameters = number of elements).

Conventions (as in klrw): an element is  sum  coeff * E_left * word,  coeff a polynomial in
the dots of the LEFT state (x_{c}_{n} = n-th strand of colour c counted from the left) and
the parameters, placed before the braid; the word acts letter by letter, first letter first.
RIGHT action on V_s = R[y_p : p a black position of s]:
    f . x_p      = y_p f
    f . psi_i    = r_c (f - s_i f)/(y_i - y_{i+1})                if s_i = s_{i+1} = c black
                 = s_i f                                          if s_i < s_{i+1}
                 = Q_{t}(y_i, y_{i+1}) * s_i f                    if s_i > s_{i+1}
where in the last case Q_t is the bigon polynomial of the TARGET state t = s_i s, and the
total order on colours is 1 < 2 < 3 < ... < W (red last).
"""
import itertools
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alg import Alg   # noqa: E402
from sage.all import PolynomialRing, ZZ, GF, matrix   # noqa: E402

ADJ = {frozenset((1, 2)), frozenset((2, 3))}
PARAMS = ["r1", "r2", "r3", "t12", "t21", "t23", "t32", "u1", "u2", "u3"]


def is_red(a):
    return isinstance(a, str)


def key(a):
    return 100 if is_red(a) else a


class PolyRep:
    def __init__(self, n, base=ZZ, values=None):
        """n = number of strands (positions).  values: dict param -> element of base (specialise)."""
        self.n = n
        if values is None:
            self.R = PolynomialRing(base, PARAMS + ["y%d" % (i + 1) for i in range(n)])
            self.p = {name: self.R(name) for name in PARAMS}
        else:
            self.R = PolynomialRing(base, ["y%d" % (i + 1) for i in range(n)])
            self.p = {name: self.R(values[name]) for name in PARAMS}
        self.y = [self.R("y%d" % (i + 1)) for i in range(n)]

    def r(self, c):
        return self.p["r%d" % c]

    def t(self, a, b):
        return self.p["t%d%d" % (a, b)]

    def u(self, c):
        return self.p["u%d" % c]

    def swap(self, f, i):
        """s_i f : exchange y_i, y_{i+1} (1-based i)."""
        y = self.y
        return f.subs({y[i - 1]: y[i], y[i]: y[i - 1]})

    def Q(self, s, i):
        """bigon polynomial of the state s at positions i, i+1."""
        a, b = s[i - 1], s[i]
        yi, yj = self.y[i - 1], self.y[i]
        if not is_red(a) and not is_red(b):
            if a == b:
                return self.R(0)
            if frozenset((a, b)) in ADJ:
                return self.t(a, b) * yi + self.t(b, a) * yj
            return self.R(1)
        if is_red(b):
            return self.u(a) * yi if a == int(b[1:]) else self.R(1)
        return self.u(b) * yj if b == int(a[1:]) else self.R(1)

    def psi(self, s, i, f):
        """f . psi_i  for f in V_s; returns (new state, polynomial)."""
        a, b = s[i - 1], s[i]
        t = list(s)
        t[i - 1], t[i] = b, a
        t = tuple(t)
        if not is_red(a) and a == b:
            num = f - self.swap(f, i)
            q, rem = num.quo_rem(self.y[i - 1] - self.y[i])
            assert rem == 0
            return t, self.r(a) * q
        if key(a) < key(b):
            return t, self.swap(f, i)
        return t, self.Q(t, i) * self.swap(f, i)

    def word(self, s, w, f):
        for i in w:
            s, f = self.psi(s, i, f)
        return s, f

    def dot(self, s, pos, f):
        assert not is_red(s[pos - 1])
        return s, self.y[pos - 1] * f

    # -- action of an element of the klrw implementation -----------------------------------
    def act_element(self, A, s, element, f):
        """rho(element)(f) for f in V_s, element in e(s) A e(*).  Returns dict state -> poly."""
        out = {}
        for braid, coeff in element:
            left = tuple(self._tok(x) for x in braid.left_state())
            assert left == s, (left, s)
            g = self.coeff_to_poly(A, s, coeff) * f
            t, h = self.word(s, tuple(braid.word()), g)
            out[t] = out.get(t, self.R(0)) + h
        return {t: h for t, h in out.items() if h != 0}

    @staticmethod
    def _tok(node):
        return "W%d" % node.node if node.is_framing() else node.node

    def coeff_to_poly(self, A, s, coeff):
        """Polynomial in dots (by colour index in the left state s) and parameters -> R."""
        D = A.D
        names = D.variable_names()
        res = self.R(0)
        for exp, c in coeff.iterator_exp_coeff():
            term = self.R(c)
            for pos, power in exp.sparse_iter():
                name = names[pos]
                if name.startswith("x_"):
                    _, col, idx = name.split("_")
                    col, idx = int(col), int(idx)
                    p = [q for q in range(len(s)) if s[q] == col][idx - 1]
                    term *= self.y[p] ** power
                elif name.startswith("r_"):
                    term *= self.r(int(name[2:])) ** power
                elif name.startswith("t_"):
                    _, a, b = name.split("_")
                    term *= self.t(int(a), int(b)) ** power
                elif name.startswith("u_"):
                    term *= self.u(int(name[2:])) ** power
                else:
                    raise ValueError(name)
            res += term
        return res


def seqstr(s):
    return " ".join(a if is_red(a) else str(a) for a in s)


def monomials(P, positions, deg):
    out = []
    for d in range(deg + 1):
        for combo in itertools.combinations_with_replacement(positions, d):
            m = P.R(1)
            for q in combo:
                m *= P.y[q]
            out.append(m)
    return out


def check_relations():
    fails = 0
    n = 0
    for k in (1, 2):
        alphabet = [1, 2, 3, "W%d" % k]
        for L in (2, 3):
            P = PolyRep(L)
            for s in itertools.product(alphabet, repeat=L):
                if sum(is_red(a) for a in s) > 1:
                    continue
                blacks = [q for q in range(L) if not is_red(s[q])]
                for f in monomials(P, blacks, 3):
                    if L == 2:
                        # quadratic
                        t, g = P.word(s, (1, 1), f)
                        n += 1
                        if not (t == s and g == P.Q(s, 1) * f):
                            fails += 1
                            print("FAIL Q", s, f)
                        # dot slides
                        same = not is_red(s[0]) and s[0] == s[1]
                        rr = P.r(s[0]) if same else 0
                        if not is_red(s[0]):     # x_1 psi - psi x_2 = r
                            _, g1 = P.psi(s, 1, P.y[0] * f)
                            t, g2 = P.psi(s, 1, f)
                            n += 1
                            if g1 - P.y[1] * g2 != rr * f:
                                fails += 1
                                print("FAIL x1psi", s, f)
                        if not is_red(s[1]):     # psi x_1 - x_2 psi = r
                            t, g1 = P.psi(s, 1, f)
                            _, g2 = P.psi(s, 1, P.y[1] * f)
                            n += 1
                            if P.y[0] * g1 - g2 != rr * f:
                                fails += 1
                                print("FAIL psix1", s, f)
                    else:
                        t1, g1 = P.word(s, (1, 2, 1), f)
                        t2, g2 = P.word(s, (2, 1, 2), f)
                        a, b, c = s
                        exp = 0
                        if not is_red(a) and a == c:
                            if not is_red(b) and frozenset((a, b)) in ADJ:
                                exp = P.r(a) * P.t(a, b)
                            if is_red(b) and int(b[1:]) == a:
                                exp = P.r(a) * P.u(a)
                        n += 1
                        if not (t1 == t2 and g1 - g2 == exp * f):
                            fails += 1
                            print("FAIL cubic", s, f, g1 - g2)
                        # far commutation of a dot with a crossing
                        if not is_red(s[2]):
                            _, h1 = P.psi(s, 1, P.y[2] * f)
                            _, h2 = P.psi(s, 1, f)
                            n += 1
                            if h1 != P.y[2] * h2:
                                fails += 1
    # far commutation psi_1 psi_3 on a 4-strand state
    P = PolyRep(4)
    for s in [(1, 2, 1, 2), (1, 1, 2, 2), (2, "W1", 1, 1), ("W1", 1, 1, 2)]:
        blacks = [q for q in range(4) if not is_red(s[q])]
        for f in monomials(P, blacks, 2):
            n += 1
            if P.word(s, (1, 3), f) != P.word(s, (3, 1), f):
                fails += 1
                print("FAIL far", s)
    print("polynomial representation: %d relation instances checked, %d failures" % (n, fails))
    return fails == 0


def random_elements(A, s, n_el, rng, maxlen=None):
    """random basis elements (dot monomial * basis braid) with left state s."""
    K = A.K
    st = A.state(seqstr(s))
    braids = list(K.KLRWBraid._braids_with_left_state_iter_(st))
    out = []
    for _ in range(n_el):
        b = rng.choice(braids)
        el = K.monomial(b)
        blacks = [q + 1 for q in range(len(s)) if not is_red(s[q])]
        for _ in range(rng.randint(0, 2)):
            el = A.dot(seqstr(s), rng.choice(blacks)) * el
        out.append(el)
    return out


def check_hom(n_pairs=60, seed=1):
    rng = random.Random(seed)
    fails = 0
    n = 0
    for seq in [(1, 2, 1), (1, "W1", 1), (1, 1, 2, "W1"), (2, 1, "W1", 1), (1, 2, 3, 2), (2, "W2", 1, 2, 2)]:
        A = Alg(["A", 3], seqstr(seq))
        P = PolyRep(len(seq))
        K = A.K
        states = sorted(set(itertools.permutations(seq)), key=str)
        for _ in range(n_pairs):
            s = rng.choice(states)
            (a,) = random_elements(A, s, 1, rng)
            # right state of a (single basis braid)
            (braid, _), = list(a)
            t = tuple(P._tok(x) for x in braid.right_state())
            (b,) = random_elements(A, t, 1, rng)
            ab = a * b
            blacks = [q for q in range(len(s)) if not is_red(s[q])]
            for f in monomials(P, blacks, 1):
                lhs = P.act_element(A, s, ab, f) if ab != 0 * ab else {}
                mid = P.act_element(A, s, a, f)
                rhs = {}
                for st2, g in mid.items():
                    for st3, h in P.act_element(A, st2, b, g).items():
                        rhs[st3] = rhs.get(st3, P.R(0)) + h
                rhs = {k2: v for k2, v in rhs.items() if v != 0}
                n += 1
                if lhs != rhs:
                    fails += 1
                    print("FAIL hom", seq, s, a, b, f)
    print("homomorphism check rho(a*b) = rho(b) o rho(a): %d instances, %d failures" % (n, fails))
    return fails == 0


def check_rank(D=2, Dtest=None, p=1000003, seed=7):
    rng = random.Random(seed)
    values = {name: rng.randint(2, p - 2) for name in PARAMS}
    ok = True
    for seq, pairs in [((1, 1, 2), None), ((1, "W1", 1), None), ((1, 2, 1, "W1"), 6), ((1, 1, "W1", 2), 6), ((1, 1, 1), None)]:
        A = Alg(["A", 3], seqstr(seq))
        K = A.K
        n = len(seq)
        P = PolyRep(n, base=GF(p), values=values)
        states = sorted(set(itertools.permutations(seq)), key=str)
        allpairs = [(s, t) for s in states for t in states]
        if pairs is not None:
            allpairs = rng.sample(allpairs, pairs)
        for s, t in allpairs:
            st, tt = A.state(seqstr(s)), A.state(seqstr(t))
            braids = K.KLRWBraid.braids_by_states(st, tt) if tt in K.KLRWBraid.braids_with_left_state_by_right_state(st) else ()
            blacks = [q for q in range(n) if not is_red(s[q])]
            elems = []
            for b in braids:
                for m in monomials(P, blacks, D):
                    elems.append((m, tuple(b.word())))
            if not elems:
                continue
            tests = monomials(P, blacks, Dtest if Dtest is not None else D + len(braids[0].word()) // 1 + 1)
            rows = []
            for m, w in elems:
                vec = {}
                for ti, f in enumerate(tests):
                    t2, g = P.word(s, w, m * f)
                    assert t2 == t
                    for mono, c in g.dict().items():
                        vec[(ti, mono)] = c
                rows.append(vec)
            cols = sorted(set(k2 for r in rows for k2 in r), key=str)
            idx = {c: i for i, c in enumerate(cols)}
            M = matrix(GF(p), len(rows), len(cols), {(i, idx[c]): v for i, r in enumerate(rows) for c, v in r.items()}, sparse=True)
            rk = M.rank()
            good = rk == len(elems)
            ok &= good
            print("   %-14s -> %-14s  %3d basis braids x %2d dot monomials = %4d elements, rank %4d %s"
                  % (seqstr(s), seqstr(t), len(braids), len(elems) // max(1, len(braids)), len(elems), rk, "ok" if good else "DEPENDENT"))
    print("rank check (F1, linear independence):", "all independent" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("relations", "all"):
        check_relations()
    if mode in ("hom", "all"):
        check_hom()
    if mode in ("rank", "all"):
        check_rank()


# ---------------------------------------------------------------------------------------
# Exact proof-by-computation of Proposition A.7: the operators satisfy the relations as
# IDENTITIES in the twisted group algebra  Frac(R[y]) # S_n  (not only on test vectors).
# An operator is a dict  w -> rational function c_w,  acting by  f |-> sum_w c_w * (w . f),
# where (w . f)(y_1..y_n) = f(y_{w(1)}, ..., y_{w(n)}) ... we only need the composition law
#     (F o G)_u = sum_{w v = u} F_w * w(G_v),
# with w acting on rational functions by permuting the variables.
# ---------------------------------------------------------------------------------------
def twisted_check():
    from sage.all import FractionField
    fails = 0
    n_checks = 0
    for L in (2, 3):
        P = PolyRep(L)
        K = FractionField(P.R)
        y = [K(v) for v in P.y]

        def perm_act(w, g):
            # w is a tuple: w[p] = image of position p (0-based); act on rational g by y_p -> y_{w[p]}
            num, den = g.numerator(), g.denominator()
            sub = {P.y[p]: P.y[w[p]] for p in range(L)}
            return K(num.subs(sub)) / K(den.subs(sub))

        def compose(F, G):
            """operator F o G (first G, then F)."""
            out = {}
            for w, cw in F.items():
                for v, dv in G.items():
                    u = tuple(w[v[p]] for p in range(L))
                    out[u] = out.get(u, K(0)) + cw * perm_act(w, dv)
            return {u: c for u, c in out.items() if c != 0}

        ident = tuple(range(L))

        def sw(i):
            w = list(range(L))
            w[i - 1], w[i] = w[i], w[i - 1]
            return tuple(w)

        def op_psi(s, i):
            a, b = s[i - 1], s[i]
            t = list(s)
            t[i - 1], t[i] = b, a
            t = tuple(t)
            if not is_red(a) and a == b:
                c = K(P.r(a)) / (y[i - 1] - y[i])
                return t, {ident: c, sw(i): -c}
            if key(a) < key(b):
                return t, {sw(i): K(1)}
            return t, {sw(i): K(P.Q(t, i))}

        def op_word(s, w):
            F = {ident: K(1)}
            for i in w:
                s, G = op_psi(s, i)
                F = compose(G, F)          # right action: the later letter acts after
            return s, F

        def op_dot(p):
            return {ident: y[p - 1]}

        def eq(F, G):
            keys = set(F) | set(G)
            return all(F.get(k2, K(0)) == G.get(k2, K(0)) for k2 in keys)

        def scal(c, F):
            return {k2: c * v for k2, v in F.items()}

        def plus(F, G):
            out = dict(F)
            for k2, v in G.items():
                out[k2] = out.get(k2, K(0)) + v
            return {k2: v for k2, v in out.items() if v != 0}

        for kk in (1, 2):
            for s in itertools.product([1, 2, 3, "W%d" % kk], repeat=L):
                if sum(is_red(a) for a in s) > 1:
                    continue
                if L == 2:
                    t, F = op_word(s, (1, 1))
                    n_checks += 1
                    if not eq(F, {ident: K(P.Q(s, 1))}):
                        fails += 1
                        print("FAIL quadratic", s)
                    same = not is_red(s[0]) and s[0] == s[1]
                    rr = K(P.r(s[0])) if same else K(0)
                    t, Fp = op_psi(s, 1)
                    if not is_red(s[0]):
                        # x_1 psi - psi x_2 = r :   op(psi) o op(x_1) - op(x_2) o op(psi)
                        lhs = plus(compose(Fp, op_dot(1)), scal(K(-1), compose(op_dot(2), Fp)))
                        n_checks += 1
                        if not eq(lhs, {ident: rr} if rr != 0 else {}):
                            fails += 1
                            print("FAIL x1 psi", s)
                    if not is_red(s[1]):
                        lhs = plus(compose(op_dot(1), Fp), scal(K(-1), compose(Fp, op_dot(2))))
                        n_checks += 1
                        if not eq(lhs, {ident: rr} if rr != 0 else {}):
                            fails += 1
                            print("FAIL psi x1", s)
                else:
                    t1, F1 = op_word(s, (1, 2, 1))
                    t2, F2 = op_word(s, (2, 1, 2))
                    a, b, c = s
                    exp = K(0)
                    if not is_red(a) and a == c:
                        if not is_red(b) and frozenset((a, b)) in ADJ:
                            exp = K(P.r(a) * P.t(a, b))
                        if is_red(b) and int(b[1:]) == a:
                            exp = K(P.r(a) * P.u(a))
                    n_checks += 1
                    if not (t1 == t2 and eq(plus(F1, scal(K(-1), F2)), {ident: exp} if exp != 0 else {})):
                        fails += 1
                        print("FAIL cubic", s)
                    # far dot
                    if not is_red(s[2]):
                        t, Fp = op_psi(s, 1)
                        n_checks += 1
                        if not eq(compose(Fp, op_dot(3)), compose(op_dot(3), Fp)):
                            fails += 1
        # far commutation needs 4 strands; it is clear (disjoint variables) and checked above on test vectors
    print("twisted group algebra: %d relation identities checked exactly, %d failures" % (n_checks, fails))
    return fails == 0


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "twisted":
    twisted_check()


# ---------------------------------------------------------------------------------------
# Degenerate parameters (R01_referee.md, §1.4(iv)): the polynomial representation is NOT
# faithful at r_c = 0, at t_{cc'} = t_{c'c} = 0, or at u_k = 0 -- the claimed basis acts with
# smaller rank.  (F1) still holds there, by base change (Theorem A.5(iv)).
#     sage -python proofs/checks/01_polyrep.py degenerate
# ---------------------------------------------------------------------------------------
def degenerate(p=1000003, seed=9, D=1):
    rng = random.Random(seed)
    generic = {name: rng.randint(2, p - 2) for name in PARAMS}
    cases = [("generic", {}), ("r1 = 0", {"r1": 0}), ("t12 = t21 = 0", {"t12": 0, "t21": 0}), ("u1 = 0", {"u1": 0})]
    for seq in [(1, 1, 2), (1, "W1", 1)]:
        A = Alg(["A", 3], seqstr(seq))
        K = A.K
        n = len(seq)
        states = sorted(set(itertools.permutations(seq)), key=str)
        for label, override in cases:
            values = dict(generic)
            values.update(override)
            P = PolyRep(n, base=GF(p), values=values)
            tot_el = tot_rk = 0
            for s in states:
                for t in states:
                    st, tt = A.state(seqstr(s)), A.state(seqstr(t))
                    bys = K.KLRWBraid.braids_with_left_state_by_right_state(st)
                    if tt not in bys:
                        continue
                    braids = bys[tt]
                    blacks = [q for q in range(n) if not is_red(s[q])]
                    elems = [(m, tuple(b.word())) for b in braids for m in monomials(P, blacks, D)]
                    tests = monomials(P, blacks, D + 3)
                    rows = []
                    for m, w in elems:
                        vec = {}
                        for ti, f in enumerate(tests):
                            _, g = P.word(s, w, m * f)
                            for mono, c in g.dict().items():
                                vec[(ti, mono)] = c
                        rows.append(vec)
                    cols = sorted(set(k2 for r in rows for k2 in r), key=str)
                    idx = {c: i for i, c in enumerate(cols)}
                    M = matrix(GF(p), len(rows), max(1, len(cols)),
                               {(i, idx[c]): v for i, r in enumerate(rows) for c, v in r.items()}, sparse=True)
                    tot_el += len(elems)
                    tot_rk += M.rank()
            print("   %-10s %-14s: %3d elements, rank in V %3d" % (seqstr(seq), label, tot_el, tot_rk))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "degenerate":
    degenerate()
