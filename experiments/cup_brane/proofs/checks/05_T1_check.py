"""
Checks for the proof of Theorem T1 (every colour at most twice) in proofs/05_main_sliding.md.

For a T1 word w = w'.c whose last colour c occurs once in w', at the slot a (last slot m):

 [5]   combinatorics of admissible configurations: distinct colours (5.1), adjacent colours => linked (5.3)
 [6.2] the element of a tree term does not depend on the reduced word; hypothesis (S2) holds
 [7]   (I1)  for a in I':  iota_R(beta) X = X iota_L(beta) for every term with mover a
 [8]   for a not in I':
         - every frozen term has at most one bad letter (5.4)
         - E(beta, chi) = beta(I; j, F' + (a,m)) * m_a, and the sliding identity, and the coefficient identity
         - the products t1 * t2 (t1 contains (a,m), t2 has mover a) come in families indexed by the subsets S
           of the k subtrees hanging at (a,m): 2^k products, all equal to the reduced diagram of the common
           permutation (hypotheses (S1), (S2) checked for every word), coefficients alternating with |S|
 [9]   m_a * m_a^* = P * e with P the predicted product of bigon polynomials
 [10]  the chain map equation (d' (x) 1) F = F (1 (x) d'), all four blocks

    sage -python 05_T1_check.py TYPE RANK WORD [WORD ...]
    sage -python 05_T1_check.py TYPE RANK all MINLEN MAXLEN [MAXWORDS] [SEED]
"""
import sys, importlib, itertools, random, time
from collections import defaultdict
L = importlib.import_module("05_lib")


def node_colour(S, x):
    return S.word[x - 1] if not isinstance(x, tuple) else S.word[x[0] - 1]


def linked(S, x, y):
    """Is there a link between the nodes x, y (a mover = slot number, a pair = tuple)?"""
    if not isinstance(x, tuple) and not isinstance(y, tuple):
        return False
    if not isinstance(y, tuple):
        x, y = y, x
    if not isinstance(x, tuple):                      # x mover, y pair
        return y[0] < x < y[1] and S.st.adjacent(x, y[0])
    (a, b), (a2, b2) = sorted((x, y))
    return a < a2 < b < b2 and S.st.adjacent(a, a2)


def check_word(ct, w, red=1, rng=random.Random(0), stats=None):
    stats = stats if stats is not None else defaultdict(int)
    m = len(w)
    c = w[-1]
    assert all(w.count(x) <= 2 for x in set(w)), "not a T1 word"
    S = L.Setup(ct, red, w)
    K = S.K
    X = tuple(range(m, 0, -1))
    a = w.index(c) + 1 if w.count(c) == 2 else None

    # [5], [6.2] for all vertices and all terms of w
    for h in range(1, m + 1):
        for I in itertools.combinations(range(1, m + 1), h):
            I = frozenset(I)
            ids = L.strand_ids(S, I)
            for t in S.terms(I):
                nodes = [t.j] + list(t.F)
                cols = [node_colour(S, x) for x in nodes]
                assert len(set(cols)) == len(cols)                                     # 5.1
                for x, y in itertools.combinations(nodes, 2):
                    if S.st.E and frozenset((node_colour(S, x), node_colour(S, y))) in S.st.E:
                        assert linked(S, x, y)                                         # 5.3
                el = S.beta(I, t.J, t.sigma)
                word = L.word_of(S, el)
                ok1, ok2, nd = L.check_S1_S2(S, ids, word)
                assert ok1 and ok2 and nd == 0
                final = L.crossings(ids, word)[1]
                for _ in range(2):
                    assert S.from_word(I, L.random_reduced_word(ids, final, rng)) == el  # 6.2
                stats["tree terms"] += 1
    if a is None:
        # new colour: X slides through everything
        for h in range(0, m):
            for Ip in itertools.combinations(range(1, m), h):
                I = frozenset(Ip) | {m}
                for t in S.terms(I):
                    if t.j == m:
                        continue
                    assert all(m not in p for p in t.F)
                    bw = L.word_of(S, S.beta(I, t.J, t.sigma))
                    assert S.from_word(I, bw + X) == S.from_word(I, X + tuple(i + 1 for i in bw))
                    stats["new colour: exact slides"] += 1
        return stats

    r_c = S.mono((("r_%d" % c, 1),))
    for h in range(0, m):
        for Ip in itertools.combinations(range(1, m), h):
            Ip = frozenset(Ip)
            I = Ip | {m}
            ids = L.strand_ids(S, I)
            terms = S.terms(I)
            # ---------------- [7] (I1) and [9]
            if a in Ip:
                for t in terms:
                    if t.j != a:
                        continue
                    bw = L.word_of(S, S.beta(I, t.J, t.sigma))
                    assert S.from_word(I, bw + X) == S.from_word(I, X + tuple(i + 1 for i in bw))
                    stats["(I1) exact slides"] += 1
                # [9]: m_a m_a^* at the vertex I' (extra strand on the far left): vertex Ip of w
                ta = [t for t in S.terms(Ip) if t.j == a and not t.F][0]
                wa = L.word_of(S, S.beta(Ip, ta.J, ta.sigma))
                lhs = S.from_word(Ip, wa + tuple(reversed(wa)))
                P = K.idempotent(S.state(Ip))
                idsp = L.strand_ids(S, Ip)
                for (x, y) in L.crossings(idsp, wa)[0]:
                    assert y == a
                    if x == "W":
                        if c == red:
                            P = P * (S.mono((("u_%d" % c, 1),)) * L.dot(S, Ip, a))
                    elif S.st.adjacent(x, a):
                        cx = w[x - 1]
                        P = P * (S.mono((("t_%d_%d" % (c, cx), 1),)) * L.dot(S, Ip, a)
                                 + S.mono((("t_%d_%d" % (cx, c), 1),)) * L.dot(S, Ip, x))
                    else:
                        assert w[x - 1] != c
                assert lhs == P and not P.is_zero()
                stats["[9] m_a m_a^* = P"] += 1
                continue
            # ---------------- [8]  a not in I'
            # frozen terms and their bad letters
            frozen = {}
            for t in terms:
                if t.j == m or any(m in p for p in t.F):
                    continue
                el = S.beta(I, t.J, t.sigma)
                bw = L.word_of(S, el)
                cr, _ = L.crossings(ids, bw)
                bad = [k for k, (s, s2) in enumerate(cr)
                       if s == a and s2 != "W" and S.st.adjacent(a, s2)]
                assert len(bad) <= 1                                                   # 5.4
                frozen[t.j, t.F] = (t, bw, bad)
                lhs = S.from_word(I, bw + X) - S.from_word(I, X + tuple(i + 1 for i in bw))
                if not bad:
                    assert lhs.is_zero()
                    stats["frozen terms without bad letter"] += 1
                    continue
                k = bad[0]
                i = bw[k]
                s2 = cr[k][1]
                Eword = bw[:k] + tuple(range(m, i + 1, -1)) + tuple(range(i - 1, 0, -1)) \
                    + tuple(x + 1 for x in bw[k + 1:])
                ok1, ok2, nd = L.check_S1_S2(S, ids, Eword)
                assert ok1 and ok2
                stats["E diagrams with %d double crossings" % nd] += 1
                Eel = S.from_word(I, Eword)
                rt = r_c * S.mono((("t_%d_%d" % (c, w[s2 - 1]), 1),))
                assert lhs == rt * Eel                                                 # sliding lemma
                # the product beta(I; j, F' + (a,m)) * m_a
                F1 = tuple(sorted(t.F + ((a, m),)))
                (t1,) = [x for x in terms if x.j == t.j and x.F == F1]
                (t2,) = [x for x in S.terms(t1.J) if x.j == a and not x.F]
                assert t2.J == t.J - {m}
                assert S.beta(I, t1.J, t1.sigma) * S.beta(t1.J, t2.J, t2.sigma) == Eel  # Lemma 8.1 (k = 0)
                sig = list(range(1, m + 1)); sig[a - 1], sig[m - 1] = m, a
                for (p, q) in t.F:
                    sig[p - 1], sig[q - 1] = q, p
                assert S.beta(I, t2.J, tuple(sig)) == Eel
                assert S.el(I, t1) * S.el(t1.J, t2) == -t.sign * S.mono(t.mono) * rt * Eel   # coefficient
                # (the sign of t as a term of w is minus its sign as a term of w')
                stats["E = product, with coefficient"] += 1
            # products t1 * t2
            families = defaultdict(list)
            for t1 in terms:
                if (a, m) not in t1.F:
                    continue
                # component of the mover after removing (a, m)
                nodes = [p for p in t1.F if p != (a, m)]
                comp = {t1.j}
                grow = True
                while grow:
                    grow = False
                    for p in nodes:
                        if p not in comp and any(linked(S, p, x) for x in comp):
                            comp.add(p); grow = True
                Fp = tuple(sorted(p for p in nodes if p in comp))
                TS = tuple(sorted(p for p in nodes if p not in comp))
                assert (t1.j, Fp) in frozen and frozen[t1.j, Fp][2]
                w1 = L.word_of(S, S.beta(I, t1.J, t1.sigma))
                for t2 in S.terms(t1.J):
                    if t2.j != a:
                        continue
                    T = tuple(sorted(TS + t2.F))
                    # union configuration: distinct colours, admissible at I
                    U = tuple(sorted(t1.F + t2.F))
                    cols = [node_colour(S, x) for x in (t1.j,) + U]
                    assert len(set(cols)) == len(cols)
                    assert any(x.j == t1.j and x.F == U for x in terms)
                    w2 = L.word_of(S, S.beta(t1.J, t2.J, t2.sigma))
                    ok1, ok2, nd = L.check_S1_S2(S, ids, w1 + w2)
                    assert ok1 and ok2
                    stats["products with %d double crossings" % nd] += 1
                    sig = list(range(1, m + 1))
                    for (p, q) in U:
                        sig[p - 1], sig[q - 1] = q, p
                    prod = S.beta(I, t1.J, t1.sigma) * S.beta(t1.J, t2.J, t2.sigma)
                    assert prod == S.beta(I, t2.J, tuple(sig))                          # Lemma 8.1
                    coef = t1.sign * t2.sign * S.mono(t1.mono) * S.mono(t2.mono)
                    # number of components of T (roots = pairs linked to (a,m))
                    roots = [q for q in T if linked(S, q, (a, m))]
                    nS = sum(1 for q in roots if q in t1.F)
                    families[t1.j, Fp, T].append((nS, len(roots), coef))
            for key, fam in families.items():
                k = fam[0][1]
                assert len(fam) == 2 ** k
                top = [co for nS, kk, co in fam if nS == k][0]
                for nS, kk, co in fam:
                    assert co == (-1) ** (k - nS) * top                                 # Lemma 8.2
                stats["families with k = %d" % k] += 1
    # ---------------- [10] the chain map equation, all blocks
    for h in range(0, m):
        for Ip in itertools.combinations(range(1, m), h):
            I = frozenset(Ip) | {m}
            tot = defaultdict(lambda: K.zero())
            for t1 in S.terms(I):
                x = S.el(I, t1)
                for t2 in S.terms(t1.J):
                    if m in t2.J:
                        continue
                    tot[t2.J] += x * S.el(t1.J, t2)
            for Kv, v in tot.items():
                assert v.is_zero()
                stats["chain map components"] += 1
    return stats


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    stats = defaultdict(int)
    t0 = time.time()
    if sys.argv[3] == "all":
        lo, hi = int(sys.argv[4]), int(sys.argv[5])
        maxwords = int(sys.argv[6]) if len(sys.argv) > 6 else 10 ** 9
        rng = random.Random(int(sys.argv[7]) if len(sys.argv) > 7 else 0)
        nw = 0
        for n in range(lo, hi + 1):
            words = list(L.words_T1(ct[1], n))
            if len(words) > maxwords:
                words = rng.sample(words, maxwords)
            for w in words:
                check_word(ct, w, red=rng.randint(1, ct[1]), rng=rng, stats=stats)
                nw += 1
            print("length %d: %d words done [%.0fs]" % (n, len(words), time.time() - t0), flush=True)
    else:
        nw = 0
        for ws in sys.argv[3:]:
            check_word(ct, tuple(int(x) for x in ws), stats=stats)
            nw += 1
    print("T1 checks passed for %d words of type %s" % (nw, ct))
    for k, v in sorted(stats.items()):
        print("   %-45s %d" % (k, v))
