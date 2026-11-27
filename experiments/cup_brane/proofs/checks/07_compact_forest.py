"""
Compact form of h with hanging forests (07_conjecture_H.md, section 9).

For a vertex I (m in I), a first chain element c and a forest F admissible for the one-element
chain (c), the claim is

  sum over the chains (c = c_1 < c_2 < ... ) compatible with F of h_{chain, F}(I)
     = eps(I; c, F) (-r_c) kappa_F * [Z_c * W_forb](source) * LS_c * [W_Lambda'] * beta(mid -> target)

  Lambda  = slots of colour c, not in I, in (c, m);
  Lambda' = Lambda minus the slots of F and minus the interiors (a_q, b_q) of the pairs q of F with
            a_q > c whose colour is c or adjacent to c (these are the chain elements allowed with F);
  Z_c     = prod (x_s - x_m) over the slots s of colour c, s < m, s not in Lambda + {c};
  W_forb  = prod (x_y - x_m), y in Lambda - Lambda'               (both read at the source);
  W_Lambda' = prod (x_y - x_m), y in Lambda', read after LS_c (the strand of c slid out to the far
            left; x_m: the strand m, not moved yet);
  kappa_F = prod_{p in F} r_p t_{c_p, c(conn_p)} eps_p for the one-element chain (c);
  beta(mid -> target) = basis braid of the remaining permutation (the pair exchanges and Seg_0).

The h-side is the formula of 06_lib / 07_lab (Conjecture H).  Also checked: a forest admissible for
some chain with first element c is admissible for (c), and the chains compatible with F are exactly
the subsets of Lambda'.

    sage -python 07_compact_forest.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, importlib, itertools
from collections import defaultdict
L = importlib.import_module("07_lab")
from formula import perm_sign


def compact_forest(S, I, c, F):
    m = S.m
    col = S.col(m)
    I = frozenset(I)
    par = S._hanging_parents([(c, m)], tuple(sorted(F)))
    if par is None:
        return None, None
    same = [s for s in range(1, m) if S.col(s) == col]
    Lam = [s for s in same if c < s < m and s not in I]
    slotsF = {x for p in F for x in p}
    bad = set(slotsF)
    for (a, b) in F:
        if a > c and (S.col(a) == col or S.adj(S.col(a), col)):
            bad |= {y for y in Lam if a < y < b}
    Lam1 = [y for y in Lam if y not in bad]
    Z = S.P.one()
    for s in same:
        if s not in Lam and s != c:
            Z *= S.x(s) - S.x(m)
    for y in Lam:
        if y in bad:
            Z *= S.x(y) - S.x(m)
    W = S.P.one()
    for y in Lam1:
        W *= S.x(y) - S.x(m)
    kappa = S.P.one()
    for p, conn in par.items():
        kappa *= S.r(S.col(p[0])) * S.t(S.col(p[0]), S.col(conn))
        if conn in I:
            kappa *= -1
    img = {m: c}
    for a, b in F:
        img[b] = a
    eps = perm_sign([img.get(x, x) for x in sorted(I)])
    J = frozenset(img.get(x, x) for x in I)
    sigma = list(range(1, m + 1))
    sigma[m - 1] = c
    sigma[c - 1] = m
    for a, b in F:
        sigma[a - 1], sigma[b - 1] = b, a
    src = S.std(I)
    el = {src: S.F.one()}
    el = S.mul_poly(el, eps * (-S.r(col)) * kappa * Z, src)
    pc = src.index(c) + 1
    el = S.word_expand(src, tuple(range(pc - 1, 0, -1)), start=el)          # LS_c
    mid = (c,) + tuple(x for x in src if x != c)
    el = S.mul_poly(el, W, mid)
    inv = {sigma[s - 1]: s for s in range(1, m + 1)}
    target = tuple(x if x == "W" else inv[x] for x in S.std(J))
    el = S.word_expand(src, L.basis_braid(list(mid), list(target)), start=el)
    return el, Lam1


def check(ct, red, word):
    S = L.Lab(ct, red, word)
    m = S.m
    n = ok = nf = 0
    chain_ok = True
    for h in range(1, m + 1):
        for rest in itertools.combinations(range(1, m), h - 1):
            I = frozenset(rest) | {m}
            groups = defaultdict(list)
            for H in S.h_terms(I):
                chain, F = H.info
                groups[(chain[0], tuple(sorted(F)))].append(H)
            for (c, F), Hs in groups.items():
                n += 1
                nf += bool(F)
                tot = defaultdict(lambda: S.F.zero())
                for H in Hs:
                    for t, f in S.expand(I, [H]).items():
                        tot[t] += f
                comp, Lam1 = compact_forest(S, I, c, F)
                if comp is None:
                    chain_ok = False
                    continue
                chains = sorted(tuple(H.info[0][1:]) for H in Hs)
                expect = sorted(tuple(x) for k in range(len(Lam1) + 1) for x in itertools.combinations(Lam1, k))
                if chains != expect:
                    chain_ok = False
                keys = set(tot) | set(comp)
                if all(tot.get(k, 0) == comp.get(k, 0) for k in keys):
                    ok += 1
    print("{} red {} word {}: compact form with forests holds for {} of {} groups (c, F) ({} with F non-empty); chains = subsets of Lambda': {}".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), ok, n, nf, chain_ok), flush=True)


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    for red, word in zip(args[0::2], args[1::2]):
        check(ct, int(red), tuple(int(x) for x in word))
