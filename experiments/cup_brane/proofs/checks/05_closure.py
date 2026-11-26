"""
Data for Section 10 of proofs/05_main_sliding.md: how far does the cocycle argument of Section 4 reach
for arbitrary non-cyclic words?

For w = w'.c the defect E (Section 3) is a cocycle (Lemma 4.1).  At lowest order in the r's the cocycle
identity at a pair (I~, K~), |K~| = |I~| - 2, relates the components

      E_{I~ - j, K~}  (j in I~, multiplied on the left by m_j)     and
      E_{I~, K~ + j}  (j not in K~, multiplied on the right by m_j),

and multiplication by m_j is injective (mod the r's) when j is the first occurrence of its colour
(Lemmas 4.2, 4.3).  So a component can be DEDUCED to vanish if, in one such identity, it is multiplied by
an m_j with j a first occurrence and all the other components of that identity are already known to
vanish.  (Proposition 4.5 is the special case "known = all components with a not in K'".)

The script computes, for random non-cyclic words in which the last colour occurs N times before:
  * the components of the equation (*) of 05_blocks_general.py that are "hard":
        MODE = dots  : some product or E has a coefficient with dots;
        MODE = clean : not clean in the sense of 05_blocks_general.py (dots, or a product that vanishes or
                       has several basis terms, or a cancellation group that is not a pair);
        MODE = dots-a1 : some product with dots contains the pair (a_1, m), a_1 = FIRST occurrence of c
                       (so the components whose dotted products only contain pairs (a_i, m), i >= 2, are
                       treated as known);
        MODE = dots-aN : some product with dots contains a pair (a_i, m) with i < N
                       (so the dotted components that only involve the pair (a_N, m), a_N = LAST occurrence
                       of c in w', are treated as known);
  * the hard components that can NOT be deduced from the others by iterating the rule above.

    sage -python 05_closure.py TYPE RANK N N_WORDS MAXLEN MODE [SEED] [shapes]

With the last argument "shapes" the script also prints, for the components that cannot be deduced, the
statistics of  (number of products with dots, number of different movers in them, number of colours of
these movers, number of different pairs containing m in them, maximal total number of pairs).
"""
import sys, importlib, itertools, random
from collections import defaultdict
L = importlib.import_module("05_lib")
from cyclefree import find_cycle
from formula import Stack


def hard_components(ct, w, mode, red=1):
    S = L.Setup(ct, red, w)
    m = S.m
    c = w[-1]
    r_c = S.mono((("r_%d" % c, 1),))
    occ = [x for x in range(1, m) if w[x - 1] == c]
    hard = set()
    for h in range(0, m):
        for Ip in itertools.combinations(range(1, m), h):
            I = frozenset(Ip) | {m}
            ids = L.strand_ids(S, I)
            groups = defaultdict(list)
            bad = set()

            def record(K, element, pairs=()):
                d = S.describe(I, K, element)
                if len(d) != 1 and mode == "clean":
                    bad.add(K)
                for k, v in d.items():
                    if len(v.monomials()) > 1:
                        firsts = [p[0] for p in pairs if p[1] == m]
                        if mode in ("dots", "clean") or (mode == "dots-a1" and occ[0] in firsts) \
                                or (mode == "dots-aN" and any(x != occ[-1] for x in firsts)):
                            bad.add(K)
                    groups[K, k].append(v)

            for t in S.terms(I):
                if t.j == m:
                    continue
                x = S.el(I, t)
                isF1 = any(m in p for p in t.F)
                for t2 in S.terms(t.J):
                    if t2.j == m or m in t2.J:
                        continue
                    record(t2.J, x * S.el(t.J, t2), t.F + t2.F)
                if not isF1:
                    K = t.J - {m}
                    bw = L.word_of(S, S.beta(I, t.J, t.sigma))
                    cr, _ = L.crossings(ids, bw)
                    for kk, (s, s2) in enumerate(cr):
                        if s != "W" and w[s - 1] == c and s2 != "W" and S.st.adjacent(s, s2):
                            i = bw[kk]
                            Ew = bw[:kk] + tuple(range(m, i + 1, -1)) + tuple(range(i - 1, 0, -1)) \
                                + tuple(y + 1 for y in bw[kk + 1:])
                            coef = r_c * S.mono((("t_%d_%d" % (c, w[s2 - 1]), 1),)) * S.mono(t.mono) * S.D(t.sign)
                            record(K, coef * S.from_word(I, Ew))
            for (K, k), g in groups.items():
                assert sum(g) == 0
                if len(g) != 2 and mode == "clean":
                    bad.add(K)
            for K in bad:
                hard.add((frozenset(Ip), K))
    return hard


def closure(w, hard):
    """The hard components that cannot be deduced."""
    n = len(w) - 1
    first = [s for s in range(1, n + 1) if w[s - 1] not in w[:s - 1]]
    unknown = set(hard)
    changed = True
    while changed and unknown:
        changed = False
        for (I, K) in list(unknown):
            ok = False
            for s in first:
                if s in K:                       # identity at (I, K - s): E_{I,K} m_s
                    Kt = K - {s}
                    nb = [(I - {j}, Kt) for j in I] + \
                         [(I, Kt | {j}) for j in range(1, n + 1) if j not in Kt and j != s]
                    if not any(x in unknown for x in nb):
                        ok = True
                        break
                if s not in I:                   # identity at (I + s, K): m_s E_{I,K}
                    It = I | {s}
                    nb = [(It - {j}, K) for j in It if j != s] + \
                         [(It, K | {j}) for j in range(1, n + 1) if j not in K]
                    if not any(x in unknown for x in nb):
                        ok = True
                        break
            if ok:
                unknown.discard((I, K))
                changed = True
    return unknown


def shape(ct, w, comp, red=1):
    S = L.Setup(ct, red, w)
    m = S.m
    Ip, K = comp
    I = Ip | {m}
    dotted = []
    for t in S.terms(I):
        if t.j == m:
            continue
        x = S.el(I, t)
        for t2 in S.terms(t.J):
            if t2.j == m or t2.J != K:
                continue
            d = S.describe(I, K, x * S.el(t.J, t2))
            if any(len(v.monomials()) > 1 for v in d.values()):
                dotted.append((t, t2))
    movers = set(j for t, t2 in dotted for j in (t.j, t2.j))
    return (len(dotted), len(movers), len(set(w[j - 1] for j in movers)),
            len(set(p for t, t2 in dotted for p in t.F + t2.F if m in p)),
            max(len(t.F) + len(t2.F) for t, t2 in dotted))


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    N = int(sys.argv[3])
    n_words, maxlen = int(sys.argv[4]), int(sys.argv[5])
    mode = sys.argv[6]
    rng = random.Random(int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != "shapes" else 0)
    seen = set()
    shapes = defaultdict(int)
    tries = tot_h = tot_u = nfail = 0
    while len(seen) < n_words and tries < 100000:
        tries += 1
        n = rng.randint(N + 2, maxlen)
        w = tuple(rng.randint(1, ct[1]) for _ in range(n))
        if w in seen or w.count(w[-1]) != N + 1 or find_cycle(Stack(ct, w), max_pairs=4) is not None:
            continue
        seen.add(w)
        hard = hard_components(ct, w, mode)
        rest = closure(w, hard)
        tot_h += len(hard)
        tot_u += len(rest)
        if sys.argv[-1] == "shapes":
            for comp in rest:
                shapes[shape(ct, w, comp)] += 1
        if rest:
            nfail += 1
            if nfail <= 6:
                print("  ", "".join(map(str, w)), ": hard", len(hard), ", not deducible", len(rest), ", e.g.",
                      [(sorted(I), sorted(K)) for I, K in sorted(rest, key=lambda x: (len(x[0]), sorted(x[0]), sorted(x[1])))[:2]], flush=True)
    print("type %s, N = %d, mode %s: %d words; %d hard components, %d of them not deducible; %d words with a remainder"
          % (ct, N, mode, len(seen), tot_h, tot_u, nfail))
    if shapes:
        print("shapes (dotted products, movers, mover colours, pairs with m, max pairs): count")
        for k, v in sorted(shapes.items()):
            print("   ", k, ":", v)
