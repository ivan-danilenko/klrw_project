"""
(star) split into groups (07_conjecture_H.md, sections 3-5).  Every contribution of (star) at a
vertex I (m in I) gets a key:
   "diag"          Phi_m, zeta*X_m, h_chain * X_{c_1}, X_j * h_chain with c_1 = j
   (j, c)          X_j or P(j;F) (tree term of the prefix with mover j) composed with a chain term
                   with first element c (either order), and zeta * P(j; F + (c, m))
For every key and every target K the sum of the expansions (in the labelled basis) is tested.
Output: number of (I, K, key) groups that vanish / do not vanish, by kind of mover j
("same" = colour of m, "adj" = colour adjacent to it, "far" = other) and by a < j / a > j.

    sage -python 07_groups.py TYPE RANK RED WORD [RED WORD ...] [v]
"""
import sys, importlib, itertools
from collections import defaultdict
L = importlib.import_module("07_lab")


def kind(S, j):
    c = S.col(S.m)
    if S.col(j) == c:
        return "same"
    return "adj" if S.adj(S.col(j), c) else "far"


ADJ_BY_J = "adjj" in sys.argv


def regroup(S, key):
    if ADJ_BY_J and key != "diag" and kind(S, key[0]) == "adj":
        return ("J", key[0])
    return key


def groups(S, I):
    m = S.m
    I = frozenset(I)
    out = defaultdict(list)     # (K, key) -> list of (label, prods, sign)
    out[(I - {m}, "diag")].append(("Phi", [S.Phi_term(I, m)], +1))
    Z = L.Term(I, I, S.zeta(), (), kind="zeta")
    for T in S.tree_terms(I):
        j, F = T.info
        if m in T.tgt:
            for H in S.h_terms(T.tgt):
                c1 = H.info[0][0]
                key = "diag" if (c1 == j and not F) else (j, c1)
                out[(H.tgt, regroup(S, key))].append(("dR.h", [T, H], +1))
        else:
            if j == m:
                out[(T.tgt, "diag")].append(("zF", [Z, T], -1))
            else:
                (a,) = [p[0] for p in F if p[1] == m]
                out[(T.tgt, regroup(S, (j, a)))].append(("zF", [Z, T], -1))
    for H in S.h_terms(I):
        c1 = H.info[0][0]
        for T in S.tree_terms(H.tgt):
            j, F = T.info
            key = "diag" if (j == c1 and not F) else (j, c1)
            out[(T.tgt, regroup(S, key))].append(("h.dL", [H, T], -1))
    return out


def run(ct, red, word, verbose=False):
    S = L.Lab(ct, red, word)
    m = S.m
    stats = defaultdict(lambda: [0, 0])
    for h in range(1, m + 1):
        for rest in itertools.combinations(range(1, m), h - 1):
            I = frozenset(rest) | {m}
            for (K, key), lst in groups(S, I).items():
                tot = defaultdict(lambda: S.F.zero())
                for label, prods, sgn in lst:
                    for t, f in S.expand(I, prods).items():
                        tot[t] += sgn * f
                ok = all(v == 0 for v in tot.values())
                if key == "diag":
                    cls = "diag"
                elif isinstance(key, tuple) and key[0] == "J":
                    cls = "adj-by-j"
                else:
                    j, a = key
                    cls = "{}, a{}j".format(kind(S, j), "<" if a < j else ">")
                stats[cls][0 if ok else 1] += 1
                if verbose and not ok:
                    print("   nonzero group", L.fmt(I), "->", L.fmt(K), key, [(lb, p) for lb, p, s in lst])
    print("{} red {} word {}: ".format("".join(map(str, ct)), red, "".join(map(str, word))) +
          "; ".join("{}: {} zero / {} nonzero".format(k, v[0], v[1]) for k, v in sorted(stats.items())), flush=True)


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    v = "v" in args
    args = [a for a in args if a not in ("v", "adjj")]
    for red, word in zip(args[0::2], args[1::2]):
        run(ct, int(red), tuple(int(c) for c in word), v)
