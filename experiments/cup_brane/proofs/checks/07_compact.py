"""
Compact form of the chain part of h (07_conjecture_H.md, Proposition 2.3):

   sum over the chains (c = c_1 < c_2 < ... < c_s) with first element c of h_chain(I)
      =  eps(I; c) * (-r_c) * Z_c(source) * LS_c * W_c * Seg_0       (see compact_expand)

   Z_c = prod_{a in Same, a not in L cap [c, m)} (x_a - x_m)      (dots at the source, slots of I)
   W_c = prod_{a in Same cap L cap (c, m)}   (x_a - x_m)          (read between the two slides)
   L = slots not in I.  Only the chain part of h (forests empty) is compared.

    sage -python 07_compact.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, importlib, itertools
from collections import defaultdict
L = importlib.import_module("07_lab")
from formula import perm_sign


def compact_expand(S, I, c):
    """
    eps * (-r) * Z(source) * LS_c * W * Seg_0, where LS_c = the strand of c slides from -c to -m
    (crossing the left slots in (c, m)), Seg_0 = the strand of m slides from +m to +c (crossing the
    right slots in (c, m)), and W = prod_{a in Same, a not in I, c < a < m} (x_a - x_m) is read at the
    state between the two slides (x_a: dot of the strand a, which LS_c has crossed; x_m: dot of the
    strand m, which has not moved yet).
    """
    m = S.m
    I = frozenset(I)
    same = [a for a in range(1, m) if S.col(a) == S.col(m)]
    Lset = [a for a in range(1, m) if a not in I]
    Z = S.P.one()
    for a in same:
        if not (a in Lset and c <= a):
            Z *= S.x(a) - S.x(m)
    W = S.P.one()
    for a in same:
        if a in Lset and c < a:
            W *= S.x(a) - S.x(m)
    eps = perm_sign([c if x == m else x for x in sorted(I)])
    src = S.std(I)
    el = {src: S.F.one()}
    el = S.mul_poly(el, eps * (-S.r(S.col(m))) * Z, src)
    pc = src.index(c) + 1                                  # 1-based position of c
    el = S.word_expand(src, tuple(range(pc - 1, 0, -1)), start=el)       # LS_c: c to the far left
    mid = (c,) + tuple(x for x in src if x != c)
    el = S.mul_poly(el, W, mid)
    n = len(src)
    k = sum(1 for x in I if c < x < m)
    el = S.word_expand(src, tuple(range(n - 1, n - 1 - k, -1)), start=el)  # Seg_0: m inwards to +c
    return el


def add(acc, el):
    for k, v in el.items():
        acc[k] = acc.get(k, 0) + v


def check(ct, red, word):
    S = L.Lab(ct, red, word)
    m = S.m
    n_ok = n = 0
    for h in range(1, m + 1):
        for rest in itertools.combinations(range(1, m), h - 1):
            I = frozenset(rest) | {m}
            chains = defaultdict(dict)
            for H in S.h_terms(I, hanging=False):
                c = H.info[0][0]
                add(chains[c], S.expand(I, [H]))
            for c, el in chains.items():
                comp = compact_expand(S, I, c)
                diff = dict(el)
                for k, v in comp.items():
                    diff[k] = diff.get(k, 0) - v
                n += 1
                n_ok += all(v == 0 for v in diff.values())
    print("{} red {} word {}: compact form = sum of chains for {} of {} (I, c)".format(
        "".join(map(str, ct)), red, "".join(map(str, word)), n_ok, n), flush=True)
    return n_ok == n


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    for red, word in zip(args[0::2], args[1::2]):
        check(ct, int(red), tuple(int(c) for c in word))
