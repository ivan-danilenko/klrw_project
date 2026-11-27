"""
Check of the claim "these are the only products of two tree terms from I to K" in
Proposition 10.2 of proofs/05_main_sliding.md (revision after R05_referee.md, item 1),
and of each step of its proof.  Pure combinatorics: plain  python3 05r_only_products.py ...

Setting: a word w (colours 1..r), slots a < j < j' < b with c(a) = c(b) = c, e = c(j), e' = c(j'),
  (H1) no slot strictly between a and b has colour c,
  (H2) no slot < j has colour e, no slot < j' has colour e',
a vertex I with j, j', b in I, a not in I, and K = (I - {j, j', b}) | {a}.

A *configuration* at a vertex V is (mu, F): mu in V, F a set of pairwise disjoint pairs (x < y) of
slots of equal colour with x not in V, y in V, y != mu.  NO admissibility is imposed (the proof in
the text does not use it), so the claim checked is stronger than the one in the proposition.

Asserted for every product (mu_1, F_1) . (mu_2, F_2) of two configurations from I to K:
  (a) {mu_1, mu_2} = {j, j'}                      [step 1 of the proof: (H2)]
  (b) j, j' lie in no pair of F_1, F_2            [step 1]
  (c) every slot other than j, j' lies in at most two pairs of F_1 u F_2; a slot lying in two pairs
      has one partner below it and one above it  [step 2: monotonicity]
  (d) F_1 u F_2 = {(a, b)}, in exactly one factor  [step 3: (H1)]
so the products are exactly P_1, P_2, P_3, P_4.  With the option `adm TYPE RANK` it also checks,
with formula.Stack, that the four configurations are admissible when e ~ c, e' ~ c (so that the
four products exist as products of tree terms).

Controls (hypotheses dropped one at a time; then extra products must occur for some words):
  noH1: (H2) kept, (H1) dropped  -> products with longer chains of pairs (a,y),(y,b), ...
  noH2: (H1) kept, (H2) dropped  -> products with other movers or pairs ending at j or j'.

    python3 05r_only_products.py COLOURS MAXLEN [adm TYPE RANK]
"""
import sys, os, itertools
from collections import Counter


def configurations(w, V):
    """All configurations (mu, F) at the vertex V (no admissibility), with their targets."""
    n = len(w)
    out = []
    for mu in sorted(V):
        avail = [(x, y) for x in range(1, n + 1) for y in range(x + 1, n + 1)
                 if w[x - 1] == w[y - 1] and x not in V and y in V and y != mu]

        def grow(start, used, F):
            out.append((mu, tuple(F), frozenset((V - {mu} - {y for x, y in F}) | {x for x, y in F})))
            for k in range(start, len(avail)):
                x, y = avail[k]
                if x in used or y in used:
                    continue
                grow(k + 1, used | {x, y}, F + [(x, y)])
        grow(0, frozenset(), [])
    return out


def products(w, I, K):
    res = []
    for mu1, F1, J in configurations(w, I):
        for mu2, F2, K2 in configurations(w, J):
            if K2 == K:
                res.append((mu1, F1, mu2, F2))
    return res


def quadruples(w, h1=True, h2=True):
    n = len(w)
    for a, j, jp, b in itertools.combinations(range(1, n + 1), 4):
        c, e, ep = w[a - 1], w[j - 1], w[jp - 1]
        if w[b - 1] != c or e == c or ep == c or e == ep:
            continue
        if h1 and c in w[a:b - 1]:
            continue
        if h2 and (e in w[:j - 1] or ep in w[:jp - 1]):
            continue
        yield a, j, jp, b


def check_product(w, a, j, jp, b, prod):
    mu1, F1, mu2, F2 = prod
    assert {mu1, mu2} == {j, jp}, ("(a)", w, prod)
    slots = Counter(x for p in F1 + F2 for x in p)
    assert slots[j] == 0 and slots[jp] == 0, ("(b)", w, prod)
    for s, d in slots.items():
        assert d <= 2, ("(c)", w, prod)
        if d == 2:
            partners = [p[0] if p[1] == s else p[1] for p in F1 + F2 if s in p]
            assert min(partners) < s < max(partners), ("(c) monotone", w, prod)
    assert sorted(F1 + F2) == [(a, b)], ("(d)", w, prod)


def main():
    r, maxlen = int(sys.argv[1]), int(sys.argv[2])
    adm = None
    if len(sys.argv) > 3 and sys.argv[3] == "adm":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        from formula import Stack
        adm = [sys.argv[4], int(sys.argv[5])]
    stats = Counter()
    ctrl = {"noH1": Counter(), "noH2": Counter()}
    example = {}
    for n in range(4, maxlen + 1):
        for w in itertools.product(range(1, r + 1), repeat=n):
            st = Stack(adm, w) if adm else None
            for a, j, jp, b in quadruples(w):
                stats["quadruples"] += 1
                others = [x for x in range(1, n + 1) if x not in (a, j, jp, b)]
                for mask in range(2 ** len(others)):
                    I = frozenset([j, jp, b] + [x for k, x in enumerate(others) if mask >> k & 1])
                    K = (I - {j, jp, b}) | {a}
                    prods = products(w, I, K)
                    for p in prods:
                        check_product(w, a, j, jp, b, p)
                    expected = {(j, (), jp, ((a, b),)), (jp, ((a, b),), j, ()),
                                (j, ((a, b),), jp, ()), (jp, (), j, ((a, b),))}
                    assert set(prods) == expected and len(prods) == 4, (w, sorted(I), prods)
                    stats["vertices"] += 1
                    stats["products"] += len(prods)
                    stats["with a further slot of colour c below a"] += (w[a - 1] in w[:a - 1])
                    if st is not None and st.adjacent(a, j) and st.adjacent(a, jp):
                        assert st.analyse(jp, ((a, b),)) is not None and st.analyse(j, ((a, b),)) is not None
                        stats["admissible checks (e ~ c, e' ~ c)"] += 1
            # controls
            for name, (h1, h2) in (("noH1", (False, True)), ("noH2", (True, False))):
                for a, j, jp, b in quadruples(w, h1, h2):
                    if h1 and h2:
                        continue
                    if (not h1 and w[a - 1] not in w[a:b - 1]) or \
                       (not h2 and w[j - 1] not in w[:j - 1] and w[jp - 1] not in w[:jp - 1]):
                        continue            # the dropped hypothesis actually holds: not a control
                    others = [x for x in range(1, n + 1) if x not in (a, j, jp, b)]
                    for mask in range(2 ** len(others)):
                        I = frozenset([j, jp, b] + [x for k, x in enumerate(others) if mask >> k & 1])
                        K = (I - {j, jp, b}) | {a}
                        prods = products(w, I, K)
                        ctrl[name]["vertices"] += 1
                        extra = [p for p in prods if not (sorted(p[1] + p[3]) == [(a, b)] and {p[0], p[2]} == {j, jp})]
                        if extra:
                            ctrl[name]["vertices with extra products"] += 1
                            for p in extra:
                                ctrl[name]["extra products with %d pairs" % len(p[1] + p[3])] += 1
                                if {p[0], p[2]} != {j, jp}:
                                    ctrl[name]["extra products with another mover"] += 1
                                key = (name, len(p[1] + p[3]))
                                if key not in example:
                                    example[key] = ("".join(map(str, w)), (a, j, jp, b), sorted(I), p)
    print("Prop. 10.2, 'only products' (all configurations, no admissibility), %d colours, words of "
          "length 4..%d: all assertions (a)-(d) hold" % (r, maxlen))
    for k, v in stats.items():
        print("   %-45s %d" % (k, v))
    for name in ctrl:
        print("control %s:" % name)
        for k, v in sorted(ctrl[name].items()):
            print("   %-45s %d" % (k, v))
    for key in sorted(example):
        print("   example %s, %d pairs: word %s, (a,j,j',b) = %s, I = %s, product %s" % ((key[0], key[1]) + example[key]))


if __name__ == "__main__":
    main()
