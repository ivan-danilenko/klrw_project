"""
Checks for Theorem C of proofs/05_main_sliding.md (section 9A): one colour c occurs three times
(slots a1 < a2 < m, m the last slot), all other colours once.

For every vertex I' and every K' with a1 not in K' the script verifies that the contributions to E_{I',K'}
are exactly those listed in section 9A and that each identity holds with its coefficient:

 (C1) the elementary moves of a1 and a2 slide through X without error;
 (C2) m_{a1} * (j; (a2,m)) = (j; (a2,m)) * m_{a1}, opposite coefficients;
 (C3) for a bad letter (A_i, j) of m_j: the word E of the Sliding Lemma is reduced, satisfies (S2), and
        i = 1:            E = (j; (a1,m)) * m_{a1}
        i = 2, a2 not in I': E = (j; (a2,m)) * m_{a2}
        i = 2, a2 in I':     E = m_{a2} * (j; (a2,m)),
      and  coef(t1) coef(t2) = coef'(m_j) r_c t_{c c(j)};
 (C4) (j; (a2,m)) * m_{j'} = m_{j'} * (j; (a2,m)) with opposite coefficients if j' is not linked to (a2,m),
      and the four-term identity of Proposition 10.2 otherwise;
 and that every tree term satisfies (S2), and finally E = 0 in ALL components (i.e. d^2 = 0).

    sage -python 05_F3_check.py TYPE RANK C OTHERS [RED]     e.g.  05_F3_check.py D 4 2 134
(all words consisting of c, c and the colours OTHERS in any order, followed by c)
"""
import sys, importlib, itertools
from collections import defaultdict
L = importlib.import_module("05_lib")


def check_word(ct, w, red, stats):
    m = len(w)
    c = w[-1]
    a1, a2 = [s for s in range(1, m) if w[s - 1] == c]
    S = L.Setup(ct, red, w)
    X = tuple(range(m, 0, -1))
    r_c = S.mono((("r_%d" % c, 1),))
    for h in range(0, m):
        for Ip in itertools.combinations(range(1, m), h):
            Ip = frozenset(Ip)
            I = Ip | {m}
            ids = L.strand_ids(S, I)
            terms = S.terms(I)
            for t in terms:                                           # (S2) for every tree term
                ok1, ok2, nd = L.check_S1_S2(S, ids, L.word_of(S, S.beta(I, t.J, t.sigma)))
                assert ok2 and nd == 0
            el = lambda V, t: S.el(V, t)
            be = lambda V, t: S.beta(V, t.J, t.sigma)
            find = lambda V, j, F: [t for t in S.terms(V) if t.j == j and t.F == F][0]
            used = set()                                              # products accounted for
            # ---- sliding
            for t in terms:
                if t.j == m or t.F:
                    continue
                K = t.J - {m}
                if a1 in K:
                    continue
                bw = L.word_of(S, be(I, t))
                lhs = S.from_word(I, bw + X) - S.from_word(I, X + tuple(i + 1 for i in bw))
                cr, _ = L.crossings(ids, bw)
                bad = [k for k, (s, s2) in enumerate(cr) if s in (a1, a2) and s2 != "W" and S.st.adjacent(s, s2)]
                if t.j in (a1, a2) or not bad:
                    assert lhs.is_zero() and not bad                  # (C1) and terms without bad letters
                    stats["exact slides"] += 1
                    continue
                j = t.j
                rt = r_c * S.mono((("t_%d_%d" % (c, w[j - 1]), 1),))
                total = S.K.zero()
                for k in bad:
                    ai = cr[k][0]
                    i = bw[k]
                    Ew = bw[:k] + tuple(range(m, i + 1, -1)) + tuple(range(i - 1, 0, -1)) + tuple(x + 1 for x in bw[k + 1:])
                    ok1, ok2, nd = L.check_S1_S2(S, ids, Ew)
                    assert ok2 and nd == 0                            # reduced, (S2)
                    Eel = S.from_word(I, Ew)
                    total += rt * Eel
                    if ai == a2 and a2 in Ip:
                        t1 = find(I, a2, ())
                        t2 = find(t1.J, j, ((a2, m),))
                    else:
                        t1 = find(I, j, ((ai, m),))
                        t2 = find(t1.J, ai, ())
                    assert t2.J == K
                    assert be(I, t1) * be(t1.J, t2) == Eel
                    assert el(I, t1) * el(t1.J, t2) == -t.sign * rt * Eel      # coef'(m_j) = -t.sign
                    used.add((t1.j, t1.F, t2.j, t2.F))
                    stats["(C3) sliding error = product"] += 1
                assert lhs == total
            # ---- products not involving X
            byK = defaultdict(list)
            for t1 in terms:
                if t1.j == m:
                    continue
                for t2 in S.terms(t1.J):
                    if t2.j == m or m in t2.J or a1 in t2.J:
                        continue
                    byK[t2.J].append((t1, t2))
            for K, prods in byK.items():
                rest = [(t1, t2) for t1, t2 in prods if (t1.j, t1.F, t2.j, t2.F) not in used]
                if not rest:
                    continue
                keys = sorted((t1.j, t1.F, t2.j, t2.F) for t1, t2 in rest)
                tot = sum((el(I, t1) * el(t1.J, t2) for t1, t2 in rest), S.K.zero())
                assert tot.is_zero()
                assert all(set(t1.F + t2.F) == {(a2, m)} for t1, t2 in rest)
                movers = set(x for k in keys for x in (k[0], k[2]))
                assert len(movers) == 2
                if len(rest) == 2:
                    (t1, t2), (u1, u2) = rest
                    assert be(I, t1) * be(t1.J, t2) == be(I, u1) * be(u1.J, u2)
                    stats["(C2)" if a1 in movers else "(C4) two terms"] += 1
                else:
                    assert len(rest) == 4 and a1 not in movers and a2 not in Ip
                    j, jp = sorted(movers)
                    assert a2 < j and all(S.st.adjacent(a2, x) for x in (j, jp))
                    stats["(C4) four terms (Prop. 10.2)"] += 1
    # ---- E = 0 in all components
    for h in range(0, m):
        for Ip in itertools.combinations(range(1, m), h):
            I = frozenset(Ip) | {m}
            tot = defaultdict(lambda: S.K.zero())
            for t1 in S.terms(I):
                x = S.el(I, t1)
                for t2 in S.terms(t1.J):
                    if m not in t2.J:
                        tot[t2.J] += x * S.el(t1.J, t2)
            for v in tot.values():
                assert v.is_zero()
                stats["components of d^2 (m in I, m not in K)"] += 1


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    c = int(sys.argv[3])
    others = [int(x) for x in sys.argv[4]]
    stats = defaultdict(int)
    words = sorted(set(itertools.permutations([c, c] + others)))
    for k, perm in enumerate(words):
        red = int(sys.argv[5]) if len(sys.argv) > 5 else 1 + k % ct[1]
        check_word(ct, tuple(perm) + (c,), red, stats)
    print("Theorem C checks passed: type %s, colour %d three times, others %s once: %d words" % (ct, c, others, len(words)))
    for k, v in sorted(stats.items()):
        print("   %-45s %d" % (k, v))
