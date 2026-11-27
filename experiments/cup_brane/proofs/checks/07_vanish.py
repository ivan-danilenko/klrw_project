"""
The vanishing and pairwise statements of 07_conjecture_H.md, checked one by one (labelled
expansion of 07_lab; zero there <=> zero in A).  For every vertex I (m in I):

  (V1)  j of colour c, j in I, a not in I, a < j:   h_chain(I) X_j = 0 for every chain with c_1 = a,
        and  X_j(I) H_a(I - j) = 0  (H_a = sum of the chains with first element a).
  (V2)  j of a colour adjacent to c, pair (a, b) around j at I, chain start c' with j < c' < b:
        P(j;(a,b)) H_{c'} = 0  and  H_{c'} P(j;(a,b)) = 0.
  (V3)  pairwise: for j of colour c with a > j, for j of a colour not adjacent to c with any a, for j
        adjacent with a > j, and for P(j;(a,b)) with chain start c' > b: the two products
        (prefix term first / chain first) add up to zero (with their signs in (star)), chain by chain
        where stated in the text, else for the compact sums.

    sage -python 07_vanish.py TYPE RANK RED WORD [RED WORD ...]
"""
import sys, importlib, itertools
from collections import defaultdict
L = importlib.import_module("07_lab")
G = importlib.import_module("07_groups")


def is_zero(S, I, prods_signs):
    tot = defaultdict(lambda: S.F.zero())
    for prods, sgn in prods_signs:
        for t, f in S.expand(I, prods).items():
            tot[t] += sgn * f
    return all(v == 0 for v in tot.values())


def run(ct, red, word):
    S = L.Lab(ct, red, word)
    m = S.m
    st = defaultdict(lambda: [0, 0])
    for h in range(1, m + 1):
        for rest in itertools.combinations(range(1, m), h - 1):
            I = frozenset(rest) | {m}
            hI = S.h_terms(I)
            dI = S.tree_terms(I)
            # products  prefix-term * h   and   h * prefix-term,  indexed
            pre = defaultdict(list)    # (j, F, c1, K) -> list of (prods, sign)  for d_w(I->J) h(J->K)
            post = defaultdict(list)   # (j, F, c1, K) -> list for h(I->J) d_w(J->K)
            for T in dI:
                if m in T.tgt:
                    for H in S.h_terms(T.tgt):
                        pre[(T.info[0], T.info[1], H.info[0][0], H.tgt)].append(([T, H], +1))
            for H in hI:
                for T in S.tree_terms(H.tgt):
                    post[(T.info[0], T.info[1], H.info[0][0], T.tgt)].append(([H, T], -1))
            keys = set(pre) | set(post)
            for key in keys:
                j, F, c1, K = key
                kd = G.kind(S, j)
                if kd == "same" and c1 < j and c1 != j:
                    # (V1): each chain of the post side separately, the pre side as a whole
                    for item in post.get(key, []):
                        st["V1 h.X_j per chain"][0 if is_zero(S, I, [item]) else 1] += 1
                    if pre.get(key):
                        st["V1 X_j.H_a"][0 if is_zero(S, I, pre[key]) else 1] += 1
                elif kd == "adj" and len(F) == 1 and j < c1 < F[0][1] and c1 not in F[0]:
                    if pre.get(key):
                        st["V2 P.H"][0 if is_zero(S, I, pre[key]) else 1] += 1
                    if post.get(key):
                        st["V2 H.P"][0 if is_zero(S, I, post[key]) else 1] += 1
                elif (kd == "same" and c1 > j) or kd == "far" or (kd == "adj" and not F and c1 > j) \
                        or (kd == "adj" and len(F) == 1 and c1 > F[0][1]):
                    both = pre.get(key, []) + post.get(key, [])
                    st["V3 pairwise " + kd + ("" if not F else " F")][0 if is_zero(S, I, both) else 1] += 1
    print("{} red {} word {}: ".format("".join(map(str, ct)), red, "".join(map(str, word))) +
          "; ".join("{}: {} ok / {} fail".format(k, v[0], v[1]) for k, v in sorted(st.items())), flush=True)


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    args = sys.argv[3:]
    for red, word in zip(args[0::2], args[1::2]):
        run(ct, int(red), tuple(int(x) for x in word))
