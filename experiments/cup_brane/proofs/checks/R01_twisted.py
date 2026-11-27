"""
R01_twisted.py  (referee of 01_foundations.md; own code, does not import 01_*.py)

    sage -python proofs/checks/R01_twisted.py

Exact check, in the twisted group algebra K # S_n (K = Frac Z[params, y]), that the operators
of the polynomial representation (01, Definition A.2) satisfy every relation (A1)-(A6) of
Definition A.1 on every state of 2 and 3 strands, for

  * Gamma = D_4 (1-2, 2-3, 2-4): adjacent / non-adjacent / trivalent node,
  * red strand of colour k = 1 and k = 2,
  * three different total orders on the colours (natural with W last; reversed with W last;
    W first) -- the representation should not depend on the order.

Operators: dict  w -> c_w,  w a tuple (w[p] = new position of the variable at position p),
acting by  f |-> sum_w c_w * w(f),  w(f) = f with y_p replaced by y_{w[p]}.
Right module: the operator of "first a, then b" is Op(b) o Op(a).

Also checked: the two (A4) consequences "x_{i+1} psi = psi x_i - r", "psi x_{i+1} = x_i psi - r";
the anti-involution iota (reverse the word) maps each relation to a relation (trivial, but
checked on the operator side through the relation table).
"""
import itertools
import sys
from sage.all import PolynomialRing, ZZ, FractionField

GAMMA_NODES = [1, 2, 3, 4]
GAMMA_EDGES = {frozenset((1, 2)), frozenset((2, 3)), frozenset((2, 4))}
PAR = ["r%d" % c for c in GAMMA_NODES] + ["u%d" % c for c in GAMMA_NODES] + \
      ["t%d%d" % (a, b) for a in GAMMA_NODES for b in GAMMA_NODES if frozenset((a, b)) in GAMMA_EDGES]
NMAX = 3
P = PolynomialRing(ZZ, PAR + ["y%d" % p for p in range(NMAX)])
K = FractionField(P)
Y = [K(P("y%d" % p)) for p in range(NMAX)]
G = {n: K(P(n)) for n in PAR}


def adj(a, b):
    return frozenset((a, b)) in GAMMA_EDGES


def Qpoly(a, b, yi, yj, k):
    """bigon polynomial of the state (a, b) (left, right), dots yi (left), yj (right)."""
    if a == "W" and b == "W":
        raise ValueError
    if a == "W":
        return G["u%d" % b] * yj if b == k else K(1)
    if b == "W":
        return G["u%d" % a] * yi if a == k else K(1)
    if a == b:
        return K(0)
    if adj(a, b):
        return G["t%d%d" % (a, b)] * yi + G["t%d%d" % (b, a)] * yj
    return K(1)


def Cval(s, k):
    a, b, c = s
    if a == c and a != "W":
        if b != "W" and adj(a, b):
            return G["r%d" % a] * G["t%d%d" % (a, b)]
        if b == "W" and a == k:
            return G["r%d" % a] * G["u%d" % a]
    return K(0)


def act(w, g, n):
    sub = {P("y%d" % p): P("y%d" % w[p]) for p in range(n)}
    num, den = g.numerator(), g.denominator()
    return K(num.subs(sub)) / K(den.subs(sub))


def compose(F, Gop, n):
    """F o G  (first G, then F)."""
    out = {}
    for w, cw in F.items():
        for v, dv in Gop.items():
            u = tuple(w[v[p]] for p in range(n))
            out[u] = out.get(u, K(0)) + cw * act(w, dv, n)
    return {u: c for u, c in out.items() if c != 0}


def add(F, Gop, sc=1):
    out = dict(F)
    for w, c in Gop.items():
        out[w] = out.get(w, K(0)) + sc * c
    return {w: c for w, c in out.items() if c != 0}


def ident(n):
    return tuple(range(n))


def transp(i, n):
    w = list(range(n))
    w[i], w[i + 1] = w[i + 1], w[i]
    return tuple(w)


def op_psi(s, i, k, rank):
    """operator of psi_{i,s} (0-based i: positions i, i+1); returns (target, operator)."""
    n = len(s)
    a, b = s[i], s[i + 1]
    t = list(s)
    t[i], t[i + 1] = b, a
    t = tuple(t)
    if a == b and a != "W":
        c = G["r%d" % a] / (Y[i] - Y[i + 1])
        return t, {ident(n): c, transp(i, n): -c}
    if rank[a] < rank[b]:
        return t, {transp(i, n): K(1)}
    return t, {transp(i, n): Qpoly(t[i], t[i + 1], Y[i], Y[i + 1], k)}


def op_word(s, word, k, rank):
    n = len(s)
    F = {ident(n): K(1)}
    for i in word:
        s, Gop = op_psi(s, i, k, rank)
        F = compose(Gop, F, n)
    return s, F


def op_dot(p, n):
    return {ident(n): Y[p]}


def scalar(c, n):
    return {ident(n): c} if c != 0 else {}


def main():
    orders = {
        "natural, W last": {1: 0, 2: 1, 3: 2, 4: 3, "W": 9},
        "reversed, W last": {4: 0, 3: 1, 2: 2, 1: 3, "W": 9},
        "natural, W first": {"W": -1, 1: 0, 2: 1, 3: 2, 4: 3},
    }
    n_checks = 0
    fails = []
    nonzero_C = set()
    for oname, rank in orders.items():
        for k in (1, 2):
            alph = GAMMA_NODES + ["W"]
            for n in (2, 3):
                for s in itertools.product(alph, repeat=n):
                    if s.count("W") > 1:
                        continue
                    for i in range(n - 1):
                        t, Fp = op_psi(s, i, k, rank)
                        # (A5)
                        _, F2 = op_word(s, (i, i), k, rank)
                        exp = scalar(Qpoly(s[i], s[i + 1], Y[i], Y[i + 1], k), n)
                        n_checks += 1
                        if add(F2, exp, -1):
                            fails.append(("A5", oname, k, s, i))
                        same = s[i] == s[i + 1] and s[i] != "W"
                        rr = G["r%d" % s[i]] if same else K(0)
                        # (A4): x_i psi - psi x_{i+1} = r ; psi x_i - x_{i+1} psi = r ; dots slide otherwise
                        if s[i] != "W":
                            lhs = add(compose(Fp, op_dot(i, n), n), compose(op_dot(i + 1, n), Fp, n), -1)
                            n_checks += 1
                            if add(lhs, scalar(rr, n), -1):
                                fails.append(("A4a", oname, k, s, i))
                            # consequence: psi x_{i+1} = x_i psi - r   (dot at target position i+1 = strand from i)
                        if s[i + 1] != "W":
                            lhs = add(compose(op_dot(i, n), Fp, n), compose(Fp, op_dot(i + 1, n), n), -1)
                            n_checks += 1
                            if add(lhs, scalar(rr, n), -1):
                                fails.append(("A4b", oname, k, s, i))
                        if same:
                            # x_{i+1} psi = psi x_i - r  and  psi x_{i+1} = x_i psi - r  (README remark (2))
                            l1 = add(compose(Fp, op_dot(i + 1, n), n), compose(op_dot(i, n), Fp, n), -1)
                            l2 = add(compose(op_dot(i + 1, n), Fp, n), compose(Fp, op_dot(i, n), n), -1)
                            n_checks += 2
                            if add(l1, scalar(-rr, n), -1) or add(l2, scalar(-rr, n), -1):
                                fails.append(("A4 consequences", oname, k, s, i))
                        # (A3): far dots
                        for p in range(n):
                            if p in (i, i + 1) or s[p] == "W":
                                continue
                            n_checks += 1
                            if add(compose(Fp, op_dot(p, n), n), compose(op_dot(p, n), Fp, n), -1):
                                fails.append(("A3", oname, k, s, i, p))
                    if n == 3:
                        t1, F1 = op_word(s, (0, 1, 0), k, rank)
                        t2, F2 = op_word(s, (1, 0, 1), k, rank)
                        C = Cval(s, k)
                        n_checks += 1
                        if t1 != t2 or add(add(F1, F2, -1), scalar(C, n), -1):
                            fails.append(("A6", oname, k, s))
                        if C != 0:
                            nonzero_C.add((s, k, str(C)))
    sys.stdout.flush(); print("twisted group algebra K # S_n, Gamma = D_4, red k in {1,2}, 3 total orders:")
    print("  %d relation identities checked exactly, %d failures" % (n_checks, len(fails)))
    for f in fails[:20]:
        print("  FAIL", f)
    print("  (state, red colour, C) with non-zero cubic constant:")
    for item in sorted(nonzero_C, key=str):
        print("    ", item)
    return not fails


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
