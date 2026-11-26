"""
Degrees for the omega_1 I-brane (proofs/02_omega1_ibrane.md, section 6), pure Python.

    python3 proofs/checks/02_degree.py N_MAX

Grading group (PROOFS.md (F2), = klrw/gradings with all scalings; names in klrw: d_0 = q,
d_{V1--W1} = e_0, d_{Vx--V(x+1)} = e_x, d_{Wn--Vn} = e_n, d_{Vc,d} = delta_c, d_{Vc,c} = sigma_c).
Edge x (0 <= x <= n) joins the labels x and x+1 (label 0 = W_1, label n+1 = W_n); write
E_x = q + e_x for the degree of a crossing of the two labels of edge x.

Checks:
 (a) Prop. 6.2: Delta(v) = sum_x m_x(v) E_x,  m_x(v) = o_x(v) + 2[x <= k-1] + 2[x in C],
     makes every term homogeneous: deg(coefficient) + deg(beta) = Delta(v) - Delta(w);
 (b) Delta - D_cube is constant on each envelope S_k (D_cube = the two-stack cube grading of
     PROOFS.md section 1.3 / Prop. 6);
 (c) Lemma 7.2 (*): |A_v| - |A_w| <= d - 1 + |F(v,w)| for v in S_k, w in S_l, l < k, d = k - l,
     and, when hdeg(w) = hdeg(v) - 1, equality holds iff (v, w) is a term of the formula;
 (d) Cor. 7.3: the dot degree N(v,w) = d + |C_v| - |C_w| - |F(v,w)| is negative whenever
     hdeg(w) = hdeg(v) (no homotopies), and for hdeg(w) = hdeg(v) - 1 it is >= 0 iff (v,w) is a
     formula pair, and then = 0;  also N computed from Delta directly (q-component) agrees.
"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
M = importlib.import_module("02_model")


def E(x):
    return Counter({"q": 1, "e%d" % x: 1})


def add(*cs, sign=1):
    out = Counter()
    for c in cs:
        for key, v in c.items():
            out[key] += v
    return out


def scale(c, s):
    return Counter({key: s * v for key, v in c.items()})


def clean(c):
    return {key: v for key, v in c.items() if v}


def edge_of_param(n, name):
    """Degree of a parameter: t_{a,b}: 2 e_{edge(a,b)} - delta_a ; u_1: 2 e_0 - delta_1 ; u_n: 2 e_n - delta_n."""
    if name.startswith("u_"):
        c = int(name[2:])
        if n == 1:
            return Counter({"e0": 2, "d1": -1})   # n = 1: both red strands have colour 1, one u_1
        return Counter({("e0" if c == 1 else "e%d" % n): 2, "d%d" % c: -1})
    _, a, b = name.split("_")
    a, b = int(a), int(b)
    return Counter({"e%d" % min(a, b): 2, "d%d" % a: -1})


def deg_coeff(n, mono):
    out = Counter()
    for name, e in mono:
        out = add(out, scale(edge_of_param(n, name), e))
    return out


def norm(n, c):
    """For n = 1 the symbols e0 and e1 coincide (both red strands have colour 1)."""
    if n == 1:
        c = Counter(c)
        c["e0"] += c.pop("e1", 0)
    return clean(c)


def deg_beta(n, v, w):
    ov, ow = M.order_bits(n, v), M.order_bits(n, w)
    out = Counter()
    for x in range(n + 1):
        if ov[x] != ow[x]:
            out = add(out, E(x))
    return out


def Delta(n, v):
    k, A, C = v
    o = M.order_bits(n, v)
    out = Counter()
    for x in range(n + 1):
        m = o[x] + 2 * (x <= k - 1) + 2 * (x in C)
        out = add(out, scale(E(x), m))
    return out


def D_cube(n, v):
    """Two-stack cube grading: sum_{c in B_L} E_{c-1} + sum_{c in C} E_c."""
    k, A, C = v
    out = Counter()
    for c in range(1, k + 1):
        if c not in A:
            out = add(out, E(c - 1))
    for c in C:
        out = add(out, E(c))
    return out


def F_set(n, v, w):
    ov, ow = M.order_bits(n, v), M.order_bits(n, w)
    return [x for x in range(n + 1) if ov[x] == 0 and ow[x] == 1]


def main(N):
    for n in range(1, N + 1):
        V = list(M.vertices(n))
        terms = M.all_terms(n)
        bad_hom = 0
        for t in terms:
            lhs = add(deg_coeff(n, t.coeff), deg_beta(n, t.src, t.tgt))
            rhs = add(Delta(n, t.src), scale(Delta(n, t.tgt), -1))
            if norm(n, lhs) != norm(n, rhs):
                bad_hom += 1
        consts = {}
        bad_cube = 0
        for v in V:
            diff = norm(n, add(Delta(n, v), scale(D_cube(n, v), -1)))
            key = v[0]
            if key in consts and consts[key] != diff:
                bad_cube += 1
            consts.setdefault(key, diff)
        form = set((t.src, t.tgt) for t in terms)
        bad_star = bad_eq = bad_N = 0
        npairs = 0
        for v in V:
            for w in V:
                k, l = v[0], w[0]
                if l >= k:
                    continue
                d = k - l
                Fv = F_set(n, v, w)
                lhs = len(v[1]) - len(w[1])
                if lhs > d - 1 + len(Fv):
                    bad_star += 1
                dh = M.hdeg(n, w) - M.hdeg(n, v)
                Nvw = d + len(v[2]) - len(w[2]) - len(Fv)
                # N from the q-component of Delta: 2N + #flips = q-part of Delta(v) - Delta(w)
                qpart = Delta(n, v)["q"] - Delta(n, w)["q"]
                flips = sum(1 for x in range(n + 1) if M.order_bits(n, v)[x] != M.order_bits(n, w)[x])
                if 2 * Nvw + flips != qpart:
                    bad_N += 1
                if dh == -1:
                    npairs += 1
                    eq = (lhs == d - 1 + len(Fv))
                    if eq != ((v, w) in form):
                        bad_eq += 1
                    if (Nvw >= 0) != ((v, w) in form) or ((v, w) in form and Nvw != 0):
                        bad_N += 1
                if dh == 0 and Nvw >= 0:
                    bad_N += 1
        print("n=%d: terms %d, inhomogeneous %d; Delta - D_cube non-constant on an envelope: %d; "
              "(*) violated %d; equality case != formula pairs %d (of %d pairs with hdeg -1); N-claims violated %d"
              % (n, len(terms), bad_hom, bad_cube, bad_star, bad_eq, npairs, bad_N), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)


# ----------------------------------------------------------------------------- polynomial model
def pi_mono(n, y):
    """pi_y = prod_{x=y..n} t_{x,x+1} * prod_{x=1..y-1} t_{x,x-1}   (t_{n,n+1} = u_n, t_{1,0} = u_1):
    one parameter for every edge e != y-1, namely the one at the end of e nearer to the edge y-1."""
    c = Counter()
    for x in range(y, n + 1):
        c[M.tname(n, x, x + 1)] += 1
    for x in range(1, y):
        c[M.tname(n, x, x - 1)] += 1
    return c


def check_polynomial_model(n):
    """Cor. 6.4: with f(k) = prod_{y<=k} pi_y every coefficient * f(k)/f(l) is a polynomial monomial;
    and f is the minimal per-envelope rescaling (longest-path solution with f(0) = 1)."""
    terms = M.all_terms(n)
    bad = 0
    for t in terms:
        k = t.src[0]; l = t.tgt[0]
        c = Counter(dict(t.coeff))
        for y in range(l + 1, k + 1):
            c.update(pi_mono(n, y))
        if any(v < 0 for v in c.values()):
            bad += 1
    # minimality: per variable, longest path from envelope 0 in the constraint graph a_k - a_l >= -e
    names = set()
    for t in terms:
        names |= {nm for nm, _ in t.coeff}
    for y in range(1, n + 1):
        names |= set(pi_mono(n, y))
    nonminimal = 0
    for nm in names:
        cons = [(t.src[0], t.tgt[0], -dict(t.coeff).get(nm, 0)) for t in terms if t.src[0] != t.tgt[0]]
        a = [0] + [-10 ** 9] * n
        for _ in range(n + 2):
            for k, l, lb in cons:            # a_k >= a_l + lb
                if a[l] > -10 ** 9 and a[l] + lb > a[k]:
                    a[k] = a[l] + lb
        f = [sum(pi_mono(n, y)[nm] for y in range(1, k + 1)) for k in range(n + 1)]
        if a != f:
            nonminimal += 1
    return len(terms), bad, nonminimal


if __name__ == "__main__":
    for n in range(1, (int(sys.argv[1]) if len(sys.argv) > 1 else 6) + 1):
        nt, bad, nm = check_polynomial_model(n)
        print("n=%d: polynomial model f(k) = prod pi_y: %d terms, non-polynomial %d; variables where f is not the"
              " minimal rescaling %d" % (n, nt, bad, nm), flush=True)
