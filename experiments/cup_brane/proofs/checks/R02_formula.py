"""
Referee checks R02 for proofs/02_omega1_ibrane.md -- pure Python, written from the TEXT of
GLUING.md section 2 and of 02_omega1_ibrane.md; nothing is imported from glue/ or from 02_*.py.

    python3 proofs/checks/R02_formula.py d2 NMAX            # d^2 = 0 through the bigon rule, random points mod p
    python3 proofs/checks/R02_formula.py d2exact NMAX       # same, exact rationals, symbolic in nothing (random Q points)
    python3 proofs/checks/R02_formula.py rmodel NMAX        # Prop. 4.2 / Lemma 4.3 entry by entry + no extra entries
    python3 proofs/checks/R02_formula.py WH NMAX            # Theorem 5.1 (delta^2 = W.H) with free q, exact over Z[q]
    python3 proofs/checks/R02_formula.py block DMAX         # Lemmas 5.3, 5.4 on the block [1,D], exact over Z[q]
    python3 proofs/checks/R02_formula.py grading NMAX       # Thm 6.3, Def 6.2, Lemma 7.1, Cor 7.2 with an own F2 grading
    python3 proofs/checks/R02_formula.py homology NMAX      # Thm 8.1: every b, at x = 0 and at a random x, mod p
    python3 proofs/checks/R02_formula.py uniq NMAX          # Thm 7.3: full-ansatz chain maps / homotopies in K_1, mod p

Conventions (from the text): vertex (k, s), s[c-1] in {'A','L','R','C'} (L = B_L, R = B_R);
idempotent [A decreasing] W1 [B increasing] Wn [C decreasing]; labels 0 = W1, n+1 = Wn.
Parameters: 'L%d' % x = t_{x,x-1} (L1 = u_1), 'R%d' % x = t_{x,x+1} (Rn = u_n); for n = 1, L1 = R1 = u_1.
"""
import itertools
import random
import sys
from collections import defaultdict
from fractions import Fraction

H = {'A': 1, 'C': 1, 'L': 2, 'R': 0}


# ----------------------------------------------------------------------------- the formula
def vertices(n):
    for k in range(n + 1):
        for left in itertools.product('AL', repeat=k):
            for right in itertools.product('RC', repeat=n - k):
                yield (k, left + right)


def seq(n, v):
    k, s = v
    A = [c for c in range(1, n + 1) if s[c - 1] == 'A']
    B = [c for c in range(1, n + 1) if s[c - 1] in 'LR']
    C = [c for c in range(1, n + 1) if s[c - 1] == 'C']
    return tuple(sorted(A, reverse=True) + [0] + B + [n + 1] + sorted(C, reverse=True))


def obits(n, v):
    sq = seq(n, v)
    pos = {lab: i for i, lab in enumerate(sq)}
    return tuple(1 if pos[x] < pos[x + 1] else 0 for x in range(n + 1))


def hdeg(v):
    return sum(H[x] for x in v[1])


def _set(s, ch):
    l = list(s)
    for c, st in ch.items():
        l[c - 1] = st
    return tuple(l)


def terms(n, v):
    """list of (w, tag) from the text of GLUING.md section 2"""
    k, s = v
    out = []
    for c in range(1, n + 1):
        if c <= k and s[c - 1] == 'L':
            out.append(((k, _set(s, {c: 'A'})), ('int', c)))
        if c > k and s[c - 1] == 'C':
            out.append(((k, _set(s, {c: 'R'})), ('int', c)))
    for d in range(1, k + 1):
        l = k - d
        blk = list(range(l + 1, k + 1))
        if all(s[c - 1] == 'A' for c in blk):
            ch = {c: 'C' for c in blk[:-1]}
            ch[k] = 'R'
            out.append(((l, _set(s, ch)), ('i', d)))
        if s[l] == 'L' and all(s[c - 1] == 'A' for c in blk[1:]):
            out.append(((l, _set(s, {c: 'C' for c in blk})), ('ii', d)))
            ch3 = {c: 'C' for c in blk[:-1]}
            ch3[k] = 'R'
            for c in range(1, n + 1):
                if c <= l and s[c - 1] == 'A':
                    ch4 = dict(ch3); ch4[c] = 'L'
                    out.append(((l, _set(s, ch4)), ('iii', d, c)))
                if c > k and s[c - 1] == 'R':
                    ch4 = dict(ch3); ch4[c] = 'C'
                    out.append(((l, _set(s, ch4)), ('iii', d, c)))
    return out


def pname(n, key):
    """canonical parameter name of 'L%d'/'R%d'"""
    side, x = key[0], int(key[1:])
    if side == 'L':
        return 'u1' if x == 1 else 't%d,%d' % (x, x - 1)
    return 'un' if (x == n and n > 1) else ('u1' if n == 1 else 't%d,%d' % (x, x + 1))


def addexp(a, b, s=1):
    out = dict(a)
    for kk, vv in b.items():
        out[kk] = out.get(kk, 0) + s * vv
    return {kk: vv for kk, vv in out.items() if vv}


def tau(n, x):
    return addexp({pname(n, 'L%d' % x): 1}, {pname(n, 'R%d' % x): 1}, -1)


def T(n, x):
    out = {}
    for y in range(1, x + 1):
        out = addexp(out, tau(n, y))
    return out


def coefficient(n, v, tag):
    if tag[0] in ('int', 'i'):
        return {}
    k = v[0]; d = tag[1]; l = k - d
    out = {}
    if tag[0] == 'ii':
        for x in range(l + 1, k + 1):
            out = addexp(out, tau(n, x))
    elif tag[2] <= l:
        for x in range(tag[2], l + 1):
            out = addexp(out, tau(n, x), -1)
    else:
        for x in range(l + 1, tag[2] + 1):
            out = addexp(out, tau(n, x))
    return out


# exterior-algebra signs of GLUING.md section 2
def _ord(n, g):
    return g[1] if g[0] == 'a' else 2 * n + 2 - g[1]


def mono(n, v):
    G = set()
    for c in range(1, n + 1):
        st = v[1][c - 1]
        if st in 'AL':
            G.add(('a', c))
        if st in 'LC':
            G.add(('g', c))
    return frozenset(G)


def _ctr(n, G, x):
    assert x in G, (G, x)
    return G - {x}, (-1) ** sum(1 for y in G if _ord(n, y) < _ord(n, x))


def _wdg(n, G, x):
    assert x not in G, (G, x)
    return G | {x}, (-1) ** sum(1 for y in G if _ord(n, y) < _ord(n, x))


def gsign(n, v, tag, w):
    """sign of the term by the exterior-algebra rule; also checks that the operator lands on w"""
    k = v[0]
    G = mono(n, v)
    if tag[0] == 'int':
        ops, pre = [('i', ('g', tag[1]))], 1
    else:
        d = tag[1]; l = k - d
        rho = lambda c: [('i', ('a', c)), ('e', ('g', c))]
        if tag[0] == 'i':
            ops = sum((rho(c) for c in range(l + 1, k)), []) + [('i', ('a', k))]
            pre = (-1) ** (d + 1)
        elif tag[0] == 'ii':
            ops = [('i', ('a', l + 1))] + sum((rho(c) for c in range(l + 2, k + 1)), [])
            pre = 1
        else:
            c = tag[2]
            if d == 1:
                ops = [('i', ('a', k)), ('i', ('g', k)), ('e', ('g', c))]
            else:
                ops = [('i', ('a', l + 1))] + sum((rho(x) for x in range(l + 2, k)), []) + \
                      [('i', ('a', k)), ('e', ('g', c))]
            pre = (-1) ** ((l + 1 - c) + d) if c <= l else (-1) ** (c - k)
    s = pre
    for op, g in ops:
        G, t = (_ctr if op == 'i' else _wdg)(n, G, g)
        s *= t
    assert G == mono(n, w), (v, tag, w)
    return s


def all_terms(n):
    out = {}
    for v in vertices(n):
        lst = []
        for w, tag in terms(n, v):
            lst.append((w, tag, gsign(n, v, tag, w), coefficient(n, v, tag)))
        out[v] = lst
    return out


def Dset(ov, ou, ow):
    return tuple(x for x in range(len(ov)) if ou[x] != ov[x] and ov[x] == ow[x])


# ----------------------------------------------------------------------------- d^2 = 0
def params(n):
    names = set()
    for x in range(1, n + 1):
        names.add(pname(n, 'L%d' % x)); names.add(pname(n, 'R%d' % x))
    return sorted(names)


def check_d2(n, field, trials=2, seed=1):
    rnd = random.Random(seed + 1000 * n)
    TT = all_terms(n)
    ob = {v: obits(n, v) for v in TT}
    bad = 0; comps = 0; nonzero_before = 0
    for trial in range(trials):
        if field == 'mod':
            p = 2 ** 61 - 1
            rv = lambda: rnd.randrange(1, p)
            inv = lambda a: pow(a, p - 2, p)
            red = lambda a: a % p
        else:
            p = None
            rv = lambda: Fraction(rnd.randrange(1, 10 ** 6), rnd.randrange(1, 10 ** 6)) * rnd.choice((1, -1))
            inv = lambda a: 1 / a
            red = lambda a: a
        par = {nm: rv() for nm in params(n)}
        xs = [None] + [rv() for _ in range(n)]
        def val(e):
            out = 1
            for nm, ex in e.items():
                out = red(out * (par[nm] ** ex if ex > 0 else inv(par[nm]) ** (-ex)))
            return out
        Lv = lambda x: par[pname(n, 'L%d' % x)]
        Rv = lambda x: par[pname(n, 'R%d' % x)]
        Q = []
        for x in range(n + 1):
            if x == 0:
                Q.append(red(Lv(1) * xs[1]))
            elif x == n:
                Q.append(red(Rv(n) * xs[n]))
            else:
                Q.append(red(Rv(x) * xs[x] + Lv(x + 1) * xs[x + 1]))
        cval = {}
        for v, lst in TT.items():
            for w, tag, sg, e in lst:
                cval[(v, w)] = red(sg * val(e))
        for v, lst in TT.items():
            comp = defaultdict(int); npaths = defaultdict(int)
            for u, tag1, s1, e1 in lst:
                for w, tag2, s2, e2 in TT[u]:
                    f = red(cval[(v, u)] * cval[(u, w)])
                    for x in Dset(ob[v], ob[u], ob[w]):
                        f = red(f * Q[x])
                    comp[w] = red(comp[w] + f)
                    npaths[w] += 1
            for w, val_ in comp.items():
                comps += 1
                if val_ != 0:
                    bad += 1
    nterms = sum(len(l) for l in TT.values())
    return len(TT), nterms, comps, bad


def polynomial_model(n):
    """Cor. 6.5: rescale S_k by f(k) = prod_{y<=k} pi_y; return number of terms with a negative exponent"""
    def pi(y):
        e = {}
        for x in range(y, n + 1):
            e = addexp(e, {pname(n, 'R%d' % x): 1})
        for x in range(1, y):
            e = addexp(e, {pname(n, 'L%d' % x): 1})
        return e
    f = [{}]
    for k in range(1, n + 1):
        f.append(addexp(f[-1], pi(k)))
    neg = 0; total = 0; claimed_bad = 0
    for v in vertices(n):
        k = v[0]
        for w, tag in terms(n, v):
            l = w[0]
            e = addexp(addexp(coefficient(n, v, tag), f[k]), f[l], -1)
            total += 1
            neg += any(x < 0 for x in e.values())
            # the closed form claimed in Cor. 6.5
            if tag[0] == 'int':
                cl = {}
            else:
                d = tag[1]
                Pi = {}
                for y in range(l + 2, k + 1):
                    Pi = addexp(Pi, pi(y))
                if tag[0] == 'i':
                    cl = addexp(pi(l + 1), Pi)
                elif tag[0] == 'ii':
                    cl = addexp(pi(k + 1), Pi)
                elif tag[2] <= l:
                    cl = addexp(pi(tag[2]), Pi)
                else:
                    cl = addexp(pi(tag[2] + 1), Pi)
            claimed_bad += (cl != e)
    return total, neg, claimed_bad


# ----------------------------------------------------------------------------- R-model
def popc(x):
    return bin(x).count('1')


def eps(c, R):
    b = 1 << (c - 1)
    if R & b:
        return None
    return R | b, (-1) ** popc(R & (b - 1))


def iot(c, R):
    b = 1 << (c - 1)
    if not R & b:
        return None
    return R & ~b, (-1) ** popc(R & (b - 1))


def proj(a, b, R):
    for c in range(a, b + 1):
        if not R & (1 << (c - 1)):
            return None
    return R, 1


def apply_ops(ops, R):
    """ops applied right to left; each op is a function R -> (R', sign) or None"""
    s = 1
    for op in reversed(ops):
        r = op(R)
        if r is None:
            return None
        R, t = r
        s *= t
    return R, s


E_ = lambda c: (lambda R: eps(c, R))
I_ = lambda c: (lambda R: iot(c, R))
P_ = lambda a, b: (lambda R: proj(a, b, R))


def rmodel_apply(n, k, R):
    """delta(e_k x r_R): list of (l, R', sign, q-index or None) -- Def. 4.1"""
    out = []
    for c in range(1, n + 1):
        if c <= k:
            r = eps(c, R)
            if r: out.append((k, r[0], r[1], None))
        else:
            r = iot(c, R)
            if r: out.append((k, r[0], r[1], c))
    for l in range(k):
        d = k - l
        r = apply_ops([I_(k), P_(l + 1, k)], R)
        if r: out.append((l, r[0], -r[1], l))
        r = apply_ops([E_(l + 1), P_(l + 2, k)], R)
        if r: out.append((l, r[0], (-1) ** (d + 1) * r[1], None))
        base = [I_(k), E_(l + 1), P_(l + 2, k)]
        for c in range(1, l + 1):
            r = apply_ops([I_(c)] + base, R)
            if r: out.append((l, r[0], (-1) ** (l - c) * r[1], c - 1))
        for c in range(k + 1, n + 1):
            r = apply_ops([E_(c)] + base, R)
            if r: out.append((l, r[0], -((-1) ** (c - l)) * r[1], None))
    return out


def Rset(n, v):
    R = 0
    for c in range(1, n + 1):
        if v[1][c - 1] in 'AC':
            R |= 1 << (c - 1)
    return R


def sigma(n, v):
    k, s = v
    A = [c for c in range(1, n + 1) if s[c - 1] == 'A']
    C = [c for c in range(1, n + 1) if s[c - 1] == 'C']
    BL = [c for c in range(1, n + 1) if s[c - 1] == 'L']
    a, g = len(A), len(C)
    return (-1) ** (a * (a - 1) // 2 + g * (g - 1) // 2 + sum(A) + len(BL) * g)


def phi(n, v):
    k, s = v
    e = {}
    for c in range(1, k):
        e = addexp(e, T(n, c))
    for c in range(1, n + 1):
        if s[c - 1] == 'C':
            e = addexp(e, T(n, c))
    return e


def check_rmodel(n):
    TT = all_terms(n)
    ob = {v: obits(n, v) for v in TT}
    one = tuple([1] * (n + 1))
    byR = {(v[0], Rset(n, v)): v for v in TT}
    assert len(byR) == len(TT)
    bad = 0; cnt = 0; extra = 0
    for v, lst in TT.items():
        rm = {}
        for (l, R2, sg, qi) in rmodel_apply(n, v[0], Rset(n, v)):
            assert (l, R2) not in rm
            rm[(l, R2)] = (sg, qi)
        seen = set()
        for w, tag, sg, e in lst:
            cnt += 1
            key = (w[0], Rset(n, w))
            seen.add(key)
            D1 = tuple(x for x in range(n + 1) if ob[w][x] == one[x] != ob[v][x])
            lhs_sign = sg * sigma(n, v) * sigma(n, w)
            lhs_mono = addexp(addexp(e, phi(n, v)), phi(n, w), -1)
            if key not in rm:
                bad += 1; continue
            rs, qi = rm[key]
            want_D = () if qi is None else (qi,)
            want_mono = {} if qi is None else T(n, qi)
            if rs != lhs_sign or want_D != D1 or want_mono != lhs_mono:
                bad += 1
        extra += len(set(rm) - seen)
    return cnt, bad, extra


# ----------------------------------------------------------------------------- polynomials in q
def padd(a, b, s=1):
    out = dict(a)
    for kk, vv in b.items():
        out[kk] = out.get(kk, 0) + s * vv
    return {kk: vv for kk, vv in out.items() if vv}


def pmul(a, b):
    out = defaultdict(int)
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            out[tuple(x + y for x, y in zip(e1, e2))] += c1 * c2
    return {kk: vv for kk, vv in out.items() if vv}


def qpoly(N, i, s=1):
    e = [0] * N
    if i is not None:
        e[i] = 1
    return {tuple(e): s}


def check_WH(n):
    N = n + 1
    W = {}
    for x in range(N):
        W = padd(W, qpoly(N, x, (-1) ** x))
    bad = 0; checked = 0
    for k in range(n + 1):
        for R in range(1 << n):
            first = defaultdict(dict)
            for (l, R2, sg, qi) in rmodel_apply(n, k, R):
                first[(l, R2)] = padd(first[(l, R2)], qpoly(N, qi, sg))
            sq = defaultdict(dict)
            for (l, R2), p1 in first.items():
                for (j, R3, sg, qi) in rmodel_apply(n, l, R2):
                    sq[(j, R3)] = padd(sq[(j, R3)], pmul(qpoly(N, qi, sg), p1))
            want = defaultdict(dict)
            for j in range(k):
                r = apply_ops([I_(k), E_(j + 1), P_(j + 2, k)], R)
                if r:
                    want[(j, r[0])] = padd(want[(j, r[0])], {kk: vv * r[1] * (-1) ** (j + 1) for kk, vv in W.items()})
            for key in set(sq) | set(want):
                checked += 1
                if padd(sq.get(key, {}), want.get(key, {}), -1):
                    bad += 1
    return checked, bad


# ----------------------------------------------------------------------------- block lemmas
def check_block(D):
    N = D + 1
    def lin(terms_):  # list of (R, poly)
        out = defaultdict(dict)
        for R, p in terms_:
            if p:
                out[R] = padd(out[R], p)
        return {R: p for R, p in out.items() if p}

    def op_ops(ops, coef):
        def f(vec):
            out = []
            for R, p in vec.items():
                r = apply_ops(ops, R)
                if r:
                    out.append((r[0], {kk: vv * r[1] for kk, vv in pmul(coef, p).items()}))
            return lin(out)
        return f

    def osum(*fs):
        def f(vec):
            out = []
            for g in fs:
                out += list(g(vec).items())
            return lin(out)
        return f

    def comp(f, g):  # f after g
        return lambda vec: f(g(vec))

    one = qpoly(N, None)
    q = lambda i, s=1: qpoly(N, i, s)
    N_ = lambda a, b: op_ops([I_(b), E_(a + 1), P_(a + 2, b)], one)
    A_ = lambda a, b: op_ops([I_(b), P_(a + 1, b)], q(a, -1))
    B_ = lambda a, b: op_ops([E_(a + 1), P_(a + 2, b)], {kk: vv * (-1) ** (b - a + 1) for kk, vv in one.items()})

    def C_(a, b):
        parts = []
        for c in range(1, a + 1):
            parts.append(op_ops([I_(c), I_(b), E_(a + 1), P_(a + 2, b)], q(c - 1, (-1) ** a * (-1) ** c)))
        for c in range(b + 1, D + 1):
            parts.append(op_ops([E_(c), I_(b), E_(a + 1), P_(a + 2, b)], {kk: -vv * (-1) ** a * (-1) ** c for kk, vv in one.items()}))
        return osum(*parts) if parts else (lambda vec: {})

    Zin = lambda a, b: osum(A_(a, b), B_(a, b), C_(a, b))
    d0 = osum(*[op_ops([I_(c)], q(c)) for c in range(1, D + 1)])
    dD = osum(*[op_ops([E_(c)], one) for c in range(1, D + 1)])
    neg = lambda f: (lambda vec: {R: {kk: -vv for kk, vv in p.items()} for R, p in f(vec).items()})
    sgn = lambda s, f: f if s == 1 else neg(f)
    Phi = osum(comp(d0, Zin(0, D)), comp(Zin(0, D), dD), *[comp(Zin(0, l), Zin(l, D)) for l in range(1, D)])
    Psi = osum(neg(comp(d0, N_(0, D))), comp(N_(0, D), dD),
               *[osum(comp(N_(0, l), Zin(l, D)), sgn(-((-1) ** l), comp(Zin(0, l), N_(l, D)))) for l in range(1, D)])
    W = {}
    for x in range(N):
        W = padd(W, q(x, (-1) ** x))
    rhs = op_ops([I_(D), E_(1), P_(2, D)], {kk: -vv for kk, vv in W.items()})
    bad_phi = bad_psi = 0
    for R in range(1 << D):
        vec = {R: one}
        a, b = Phi(vec), rhs(vec)
        if any(padd(a.get(x, {}), b.get(x, {}), -1) for x in set(a) | set(b)):
            bad_phi += 1
        if Psi(vec):
            bad_psi += 1
    return 1 << D, bad_phi, bad_psi


# ----------------------------------------------------------------------------- an own F2 grading
def gvec(**kw):
    return {kk: vv for kk, vv in kw.items() if vv}


def gadd(a, b, s=1):
    return addexp(a, b, s)


def Eedge(n, x):
    x = 0 if n == 1 else x       # n = 1: both red strands have colour 1, eps_0 = eps_1 = eps_{1W}
    return {'q': 1, 'e%d' % x: 1}


def par_degree(n, name):
    if name == 'u1':
        return {'e0': 2, 'd1': -1}
    if name == 'un':
        return {'e%d' % n: 2, 'd%d' % n: -1}
    a, b = map(int, name[1:].split(','))
    return {'e%d' % min(a, b): 2, 'd%d' % a: -1}


def beta_degree(n, v, w):
    sv, sw = seq(n, v), seq(n, w)
    pv = {lab: i for i, lab in enumerate(sv)}
    pw = {lab: i for i, lab in enumerate(sw)}
    deg = {}
    flipped = []
    for x in range(n + 1):
        if (pv[x] < pv[x + 1]) != (pw[x] < pw[x + 1]):
            deg = gadd(deg, Eedge(n, x)); flipped.append(x)
    # sanity: the only crossings of non-zero degree are edges (all other pairs: distinct non-adjacent colours
    # or black-red of another colour); for n = 1 the pair (W1, Wn) never flips
    assert (pv[0] < pv[n + 1]) and (pw[0] < pw[n + 1])
    return deg, flipped


def mono_degree(n, e):
    out = {}
    for nm, ex in e.items():
        out = gadd(out, {kk: vv * ex for kk, vv in par_degree(n, nm).items()})
    return out


def Delta_text(n, v):
    """Def. 6.2"""
    k, s = v
    o = obits(n, v)
    out = {}
    for x in range(n + 1):
        m = o[x] + 2 * (x <= k - 1) + 2 * (1 <= x <= n and s[x - 1] == 'C')
        out = gadd(out, {kk: vv * m for kk, vv in Eedge(n, x).items()})
    return out


def solve_elements(n, target):
    """number of (alpha, mu) with deg(x^alpha mu) = target, alpha in N^n, mu Laurent in t,u,r -- by the
    explicit chain solution of the (injective) degree map; returns (m, count) with m = |alpha|"""
    if any(target.get('s%d' % c, 0) for c in range(1, n + 1)):
        return None, 0        # sigma only in r_c: r-exponent = sigma-component; we keep it below
    tq = target.get('q', 0)
    if tq % 2 or tq < 0:
        return tq / 2, 0
    m = tq // 2
    count = 0
    for alpha in compositions(m, n):
        if chain_solvable(n, target, alpha):
            count += 1
    return m, count


def compositions(m, n):
    if n == 1:
        yield (m,)
        return
    for a in range(m + 1):
        for rest in compositions(m - a, n - 1):
            yield (a,) + rest


def chain_solvable(n, target, alpha):
    # unknown exponents L_x, R_x (x = 1..n), r_c = 0 (no sigma); equations: e_x, d_c
    if n == 1:
        # params: u1 (deg 2 e0 - d1); dot: 2q + d1
        e0 = target.get('e0', 0)
        if e0 % 2:
            return False
        u = e0 // 2
        return target.get('d1', 0) == alpha[0] - u and all(
            kk in ('q', 'e0', 'd1') or not vv for kk, vv in target.items())
    e = lambda x: target.get('e%d' % x, 0)
    dd = lambda c: target.get('d%d' % c, 0)
    if e(0) % 2:
        return False
    L = e(0) // 2
    for c in range(1, n + 1):
        # d_c: alpha_c - L_c - R_c = d_c
        Rc = alpha[c - 1] - L - dd(c)
        if c < n:
            if e(c) % 2:
                return False
            L = e(c) // 2 - Rc
        else:
            return 2 * Rc == e(n)


def check_grading(n):
    TT = all_terms(n)
    # vertex degrees by propagation along internal and (i)_1 terms (coefficient 1)
    Dl = {(0, tuple('R' * n)): {}}
    unit = [(v, w) for v, l in TT.items() for (w, tag, s, e) in l if tag[0] == 'int' or tag == ('i', 1)]
    adj = defaultdict(list)
    for v, w in unit:
        b, _ = beta_degree(n, v, w)
        adj[v].append((w, b, -1)); adj[w].append((v, b, +1))
    stack = [(0, tuple('R' * n))]
    while stack:
        v = stack.pop()
        for w, b, s in adj[v]:
            dw = gadd(Dl[v], b, s)      # Delta(w) = Delta(v) - deg beta(v,w)  (s = -1 when v -> w)
            if w in Dl:
                assert Dl[w] == dw, ('inconsistent', v, w)
            else:
                Dl[w] = dw; stack.append(w)
    assert len(Dl) == len(TT)
    # Theorem 6.3: every term homogeneous
    inhom = 0
    for v, l in TT.items():
        for w, tag, s, e in l:
            b, _ = beta_degree(n, v, w)
            if gadd(mono_degree(n, e), b) != gadd(Dl[v], Dl[w], -1):
                inhom += 1
    # Def. 6.2 up to a constant
    consts = set()
    for v in TT:
        consts.add(tuple(sorted(gadd(Dl[v], Delta_text(n, v), -1).items())))
    # Lemma 7.1 / Cor. 7.2: all pairs v in S_k, w in S_l, l < k, hdeg(w) - hdeg(v) in {0, -1}
    termpairs = set((v, w) for v, l in TT.items() for (w, tag, s, e) in l if w[0] < v[0])
    byk = defaultdict(list)
    for v in TT:
        byk[v[0]].append(v)
    stats = {'H_pairs': 0, 'H_elements': 0, 'H_max_m': None, 'C_pairs': 0, 'C_elements_nonterm': 0,
             'C_terms_with_unique_elt': 0, 'C_terms': len(termpairs), 'C_max_m': None, 'lemma71_viol': 0}
    for k in range(n + 1):
        for l in range(k):
            for v in byk[k]:
                for w in byk[l]:
                    dh = hdeg(v) - hdeg(w)
                    if dh not in (0, 1):
                        continue
                    b, flipped = beta_degree(n, v, w)
                    tgt = gadd(gadd(Dl[v], Dl[w], -1), b, -1)
                    m, cnt = solve_elements(n, tgt)
                    # Lemma 7.1 inequality, directly
                    Av = sum(1 for x in v[1] if x == 'A'); Aw = sum(1 for x in w[1] if x == 'A')
                    ov, ow = obits(n, v), obits(n, w)
                    F = sum(1 for x in range(n + 1) if ov[x] == 0 and ow[x] == 1)
                    if Av - Aw > k - l - 1 + F:
                        stats['lemma71_viol'] += 1
                    if dh == 0:
                        stats['H_pairs'] += 1; stats['H_elements'] += cnt
                        stats['H_max_m'] = m if stats['H_max_m'] is None else max(stats['H_max_m'], m)
                    else:
                        stats['C_pairs'] += 1
                        stats['C_max_m'] = m if stats['C_max_m'] is None else max(stats['C_max_m'], m)
                        if (v, w) in termpairs:
                            stats['C_terms_with_unique_elt'] += (cnt == 1 and m == 0)
                        else:
                            stats['C_elements_nonterm'] += cnt
    return dict(vertices=len(TT), inhomogeneous=inhom, delta_text_offsets=len(consts), **stats)


# ----------------------------------------------------------------------------- homology (Thm 8.1)
def rank_mod(rows, ncols, p):
    rows = [dict(r) for r in rows if r]
    rank = 0; piv = {}
    basis = []
    for r in rows:
        r = {k: v % p for k, v in r.items() if v % p}
        for pc, br in basis:
            if pc in r:
                f = r[pc]
                for k2, v2 in br.items():
                    r[k2] = (r.get(k2, 0) - f * v2) % p
                r = {k2: v2 for k2, v2 in r.items() if v2}
        if r:
            pc = min(r); f = pow(r[pc], p - 2, p)
            basis.append((pc, {k2: v2 * f % p for k2, v2 in r.items()}))
    return len(basis)


def check_homology(n, seed=5):
    p = 1000003
    rnd = random.Random(seed + n)
    TT = all_terms(n)
    ob = {v: obits(n, v) for v in TT}
    par = {nm: rnd.randrange(1, p) for nm in params(n)}
    inv = lambda a: pow(a, p - 2, p)
    def val(e):
        out = 1
        for nm, ex in e.items():
            out = out * (pow(par[nm], ex, p) if ex > 0 else pow(inv(par[nm]), -ex, p)) % p
        return out
    Lv = lambda x: par[pname(n, 'L%d' % x)]
    Rv = lambda x: par[pname(n, 'R%d' % x)]
    def Qs(xs):
        Q = []
        for x in range(n + 1):
            if x == 0: Q.append(Lv(1) * xs[1] % p)
            elif x == n: Q.append(Rv(n) * xs[n] % p)
            else: Q.append((Rv(x) * xs[x] + Lv(x + 1) * xs[x + 1]) % p)
        return Q
    realised = set(obits(n, (0, tuple('R' * n))) for _ in [0])
    # which b occur as o(s) for a state s?  (all orderings with W1 before Wn)
    labs = list(range(n + 2))
    if n <= 6:
        realised = set()
        for perm in itertools.permutations(labs):
            pos = {lab: i for i, lab in enumerate(perm)}
            if pos[0] > pos[n + 1]:
                continue
            realised.add(tuple(1 if pos[x] < pos[x + 1] else 0 for x in range(n + 1)))
    verts = list(TT)
    byh = defaultdict(list)
    for v in verts:
        byh[hdeg(v)].append(v)
    idx = {v: i for h in byh for i, v in enumerate(byh[h])}
    results = []
    for b in sorted(realised):
        for label, xs in (('x=0', [None] + [0] * n), ('random x', [None] + [rnd.randrange(1, p) for _ in range(n)])):
            Q = Qs(xs)
            mats = defaultdict(list)    # h -> rows (one per source vertex)
            for v in verts:
                row = {}
                for w, tag, s, e in TT[v]:
                    f = s * val(e) % p
                    for x in range(n + 1):
                        if ob[w][x] == b[x] != ob[v][x]:
                            f = f * Q[x] % p
                    if f:
                        row[idx[w]] = (row.get(idx[w], 0) + f) % p
                mats[hdeg(v)].append(row)
            ranks = {h: rank_mod(mats[h], len(byh.get(h - 1, [])), p) for h in byh}
            homol = {h: len(byh[h]) - ranks[h] - ranks.get(h + 1, 0) for h in byh}
            results.append((b, label, tuple(homol[h] for h in sorted(byh))))
    return results


# ----------------------------------------------------------------------------- Thm 7.3 full ansatz in K_1
def check_uniq(n, seed=11, npts=None):
    """Unknowns: every homogeneous x^alpha mu beta(v,w) (v in S_k, w in S_l, l < k, hdeg drop 1) found by the
    own grading solver (with dots, any |alpha|); homotopies: the same with hdeg drop 0.  Equations: the
    components out of S_k of (delta_{C_{k-1}} + d_{S_k} + g)^2, evaluated in K_1 (Pol_1, Lemma 2.3) at npts random
    points (x, t, u) mod p (n + 3 points; fewer than n cannot detect the W-relation).  Returns per k: (#unknowns, nullity, #homotopy unknowns)."""
    p = 2 ** 31 - 1
    rnd = random.Random(seed + n)
    if npts is None:
        npts = n + 3      # the W-relation is a linear form in x_1..x_n: at least n points are needed
    TT = all_terms(n)
    ob = {v: obits(n, v) for v in TT}
    one = tuple([1] * (n + 1))
    # vertex degrees as in check_grading
    Dl = {(0, tuple('R' * n)): {}}
    adj = defaultdict(list)
    for v, l in TT.items():
        for (w, tag, s, e) in l:
            if tag[0] == 'int' or tag == ('i', 1):
                bb, _ = beta_degree(n, v, w)
                adj[v].append((w, bb, -1)); adj[w].append((v, bb, +1))
    st = [(0, tuple('R' * n))]
    while st:
        v = st.pop()
        for w, bb, s in adj[v]:
            if w not in Dl:
                Dl[w] = gadd(Dl[v], bb, s); st.append(w)
    pts = []
    for _ in range(npts):
        par = {nm: rnd.randrange(1, p) for nm in params(n)}
        xs = [None] + [rnd.randrange(1, p) for _ in range(n)]
        pts.append((par, xs))
    inv = lambda a: pow(a, p - 2, p)
    def val(par, e):
        out = 1
        for nm, ex in e.items():
            out = out * (pow(par[nm], ex, p) if ex > 0 else pow(inv(par[nm]), -ex, p)) % p
        return out
    def Qv(par, xs, x):
        Lv = lambda y: par[pname(n, 'L%d' % y)]
        Rv = lambda y: par[pname(n, 'R%d' % y)]
        if x == 0: return Lv(1) * xs[1] % p
        if x == n: return Rv(n) * xs[n] % p
        return (Rv(x) * xs[x] + Lv(x + 1) * xs[x + 1]) % p
    def K1(par, xs, v, w, coeff):     # coeff: value of the coefficient (already evaluated)
        f = coeff
        for x in range(n + 1):
            if ob[w][x] == one[x] != ob[v][x]:
                f = f * Qv(par, xs, x) % p
        return f
    def mu_of(target, alpha):
        """the unique Laurent monomial (exponent dict) of degree target - deg x^alpha, or None"""
        if n == 1:
            return {'u1': target.get('e0', 0) // 2}
        e = lambda x: target.get('e%d' % x, 0)
        dd = lambda c: target.get('d%d' % c, 0)
        out = {}
        L = e(0) // 2
        for c in range(1, n + 1):
            out[pname(n, 'L%d' % c)] = out.get(pname(n, 'L%d' % c), 0) + L
            Rc = alpha[c - 1] - L - dd(c)
            out[pname(n, 'R%d' % c)] = out.get(pname(n, 'R%d' % c), 0) + Rc
            if c < n:
                L = e(c) // 2 - Rc
        return {kk: vv for kk, vv in out.items() if vv}
    verts_by_k = defaultdict(list)
    for v in TT:
        verts_by_k[v[0]].append(v)
    res = []
    for k in range(1, n + 1):
        unk = []    # (v, w, alpha, mu)
        hom = []
        for v in verts_by_k[k]:
            for l in range(k):
                for w in verts_by_k[l]:
                    dh = hdeg(v) - hdeg(w)
                    if dh not in (0, 1):
                        continue
                    bb, _ = beta_degree(n, v, w)
                    tgt = gadd(gadd(Dl[v], Dl[w], -1), bb, -1)
                    tq = tgt.get('q', 0)
                    if tq < 0 or tq % 2:
                        continue
                    for alpha in compositions(tq // 2, n):
                        if chain_solvable(n, tgt, alpha):
                            (unk if dh == 1 else hom).append((v, w, alpha, mu_of(tgt, alpha)))
        # fixed differential of C_{k-1} and of S_k, as maps v -> list of (w, coeff-exp, sign)
        rows = defaultdict(lambda: defaultdict(int))    # (point, v, w) -> {unknown index: value}
        for pi_, (par, xs) in enumerate(pts):
            for j, (v, u, alpha, mu) in enumerate(unk):
                a_val = K1(par, xs, v, u, val(par, mu) * eval_dots(xs, alpha, p) % p)
                # g then the differential of C_{k-1} (all terms out of u stay in envelopes < k)
                for w, tag, s, e in TT[u]:
                    f = a_val * K1(par, xs, u, w, s * val(par, e) % p) % p
                    rows[(pi_, v, w)][j] = (rows[(pi_, v, w)][j] + f) % p
        # the internal differential of S_k, then g: predecessor lists inside S_k
        pred = defaultdict(list)
        for v in verts_by_k[k]:
            for w, tag, s, e in TT[v]:
                if w[0] == k:
                    pred[w].append((v, s, e))
        for pi_, (par, xs) in enumerate(pts):
            for j, (u, w, alpha, mu) in enumerate(unk):
                a_val = K1(par, xs, u, w, val(par, mu) * eval_dots(xs, alpha, p) % p)
                for v, s, e in pred[u]:
                    f = K1(par, xs, v, u, s * val(par, e) % p) * a_val % p
                    rows[(pi_, v, w)][j] = (rows[(pi_, v, w)][j] + f) % p
        eqs = [dict(r) for r in rows.values()]
        rk = rank_mod(eqs, len(unk), p)
        # the formula itself: unknowns that are exactly a formula term with its coefficient
        res.append((k, len(unk), len(unk) - rk, len(hom)))
    return res


def eval_dots(xs, alpha, p):
    out = 1
    for c, a in enumerate(alpha, 1):
        out = out * pow(xs[c], a, p) % p
    return out


# ----------------------------------------------------------------------------- main
if __name__ == '__main__':
    what = sys.argv[1]; N = int(sys.argv[2])
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for n in range(lo, N + 1):
        if what in ('d2', 'd2exact'):
            nv, nt, comps, bad = check_d2(n, 'mod' if what == 'd2' else 'exact', trials=2 if what == 'd2' else 1)
            tot, neg, clbad = polynomial_model(n)
            print('n=%d: %d vertices, %d terms; d^2 components checked %d, non-zero %d | polynomial model: '
                  '%d terms, %d with a negative exponent, %d differing from the closed form of Cor. 6.5'
                  % (n, nv, nt, comps, bad, tot, neg, clbad), flush=True)
        elif what == 'rmodel':
            print('n=%d: terms %d, entries differing from the R-model %d, R-model entries without a term %d'
                  % ((n,) + check_rmodel(n)), flush=True)
        elif what == 'WH':
            print('n=%d: components of delta^2 checked %d, differing from W.H: %d' % ((n,) + check_WH(n)), flush=True)
        elif what == 'block':
            if n < 2:
                continue
            print('D=%d: basis vectors %d, Lemma 5.4 failures %d, Lemma 5.3 failures %d' % ((n,) + check_block(n)), flush=True)
        elif what == 'grading':
            print('n=%d:' % n, check_grading(n), flush=True)
        elif what == 'homology':
            res = check_homology(n)
            sb = tuple([1] * (n + 1))
            bad = 0
            from math import comb
            for b, label, hom in res:
                if label == 'random x' or b != sb:
                    bad += any(hom)
                else:
                    bad += (hom != tuple(comb(n, i) if i <= n else 0 for i in range(len(hom))))
            print('n=%d: %d realised vectors b = o(s), %d (b, point) cases, unexpected homology: %d; b = 1^{n+1} at x=0: %s'
                  % (n, len(res) // 2, len(res), bad, [h for b, lab, h in res if b == sb and lab == 'x=0']), flush=True)
        elif what == 'uniq':
            print('n=%d:' % n, check_uniq(n), flush=True)
