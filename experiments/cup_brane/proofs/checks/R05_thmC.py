"""
Referee check 4: Theorem C (section 9A) and Proposition 10.2.

    sage -python R05_thmC.py C   TYPE RANK MAXLEN NPROC      all words with one colour 3x, others <= 1x, every red colour:
                                                             full d^2 = 0, and the statements (C1)-(C4) for the last step
    sage -python R05_thmC.py 102 NWORDS MAXLEN SEED          Prop. 10.2 on random arbitrary words
"""
import sys, random, itertools, time
from multiprocessing import Pool
from R05_lib import *
from R05_tools import gamma
from R05_d2 import d2_vertex


def wordsC(rank, maxlen):
    for L in range(3, maxlen + 1):
        for w in itertools.product(range(1, rank + 1), repeat=L):
            cnt = sorted((w.count(x) for x in set(w)), reverse=True)
            if cnt[0] == 3 and all(v == 1 for v in cnt[1:]):
                yield w


def checkC(args):
    ct, red, word, seed = args
    rng = random.Random(seed)
    st = defaultdict(int)
    S = Setup(ct, red, word)
    M = len(word)
    # full d^2 = 0
    for size in range(2, M + 1):
        for I in itertools.combinations(range(1, M + 1), size):
            n, bad = d2_vertex(S, frozenset(I), rng)
            st["d2 products"] += n
            if bad:
                st["FAIL d2"] += 1
                print("FAIL d2", ct, red, word, I, flush=True)
    # the last step, if the last letter is the triple colour
    c = [x for x in set(word) if word.count(x) == 3][0]
    if word[-1] != c:
        return dict(st)
    m = M; n = m - 1
    a1, a2 = [s for s in range(1, m) if word[s - 1] == c]
    Xw = tuple(range(m, 0, -1))
    arrR = lambda Ip: S.arr(set(Ip) | {m})
    arrL = lambda Ip: S.arr(Ip)
    X = lambda Ip: S.el(arrR(Ip), Xw)
    Sp = Setup(ct, red, word[:-1])

    def beta(I, j, F):
        return S.el(S.arr(I), S.beta_word(I, j, F, rng))
    subsets = [frozenset(x) for k in range(n + 1) for x in itertools.combinations(range(1, n + 1), k)]
    for Ip in subsets:
        I = Ip | {m}
        # list of terms
        for t in Sp.my_terms(Ip):
            assert t.F == () or (t.F == ((a1, a2),) and a1 < t.j < a2 and S.adj(S.col(t.j), c))
        for t in S.my_terms(I):
            if any(m in p for p in t.F):
                assert len(t.F) == 1 and t.F[0][0] in (a1, a2) and t.F[0][0] < t.j < m and S.adj(S.col(t.j), c)
        for j in sorted(Ip):
            # slide of m_j
            w_ = rand_reduced(Sp.arr(Ip), Sp.target_arr(Ip, j, ()), rng)
            Kp = Ip - {j}
            lhs = S.el(arrR(Ip), w_) * X(Kp) - X(Ip) * S.el(arrL(Ip), tuple(i + 1 for i in w_))
            _, letters = crossing_counts(Sp.arr(Ip), w_)
            bad = [(i, l, r_) for i, (l, r_) in enumerate(letters) if gamma(S, c, l, r_) is not None]
            if j in (a1, a2):
                st["C1 cases"] += 1
                assert not bad and lhs.is_zero()
                continue
            if a1 in Ip:
                continue
            # (C3)
            expect = [i for i, ai in ((1, a1), (2, a2)) if ai < j and S.adj(S.col(j), c)]
            assert sorted(l for _, l, _ in bad) == sorted({1: a1, 2: a2}[i] for i in expect) and all(r_ == j for _, _, r_ in bad)
            g = S.r(c) * S.t(c, S.col(j))
            tot = 0 * lhs
            for i_t, left, _ in bad:
                it = w_[i_t]
                Et = w_[:i_t] + tuple(range(m, it + 1, -1)) + tuple(range(it - 1, 0, -1)) + tuple(i + 1 for i in w_[i_t + 1:])
                ncnt, _l = crossing_counts(arrR(Ip), Et)
                assert all(v <= 1 for v in ncnt.values())                 # E_chi is reduced
                assert check_S1_S2(S, arrR(Ip), Et) == (True, True)
                ai = left
                if ai == a1 or a2 not in Ip:
                    # G^i_j . m_{a_i}
                    t1 = S.analyse(I, j, ((ai, m),)); t2 = S.analyse(t1.J, ai, ())
                    prod = beta(I, j, ((ai, m),)) * beta(t1.J, ai, ())
                else:
                    # m_{a_2} . G^2_j
                    t1 = S.analyse(I, a2, ()); t2 = S.analyse(t1.J, j, ((a2, m),))
                    prod = beta(I, a2, ()) * beta(t1.J, j, ((a2, m),))
                assert t1 is not None and t2 is not None and t2.J == Kp
                assert S.el(arrR(Ip), Et) == prod, ("C3 diagram", word, sorted(Ip), j, ai)
                tp = Sp.analyse(Ip, j, ())
                assert S.coef(t1) * S.coef(t2) == S.coef(tp) * g, ("C3 coef", word, sorted(Ip), j, ai)
                tot = tot + g * prod
                st["C3 sliding errors"] += 1
            assert lhs == tot
        # (C2)
        if a1 in Ip and a2 not in Ip:
            for j in sorted(Ip):
                if a2 < j and S.adj(S.col(j), c):
                    p1 = beta(I, a1, ()) * beta(I - {a1}, j, ((a2, m),))
                    p2 = beta(I, j, ((a2, m),)) * beta((I - {j, m}) | {a2}, a1, ())
                    assert p1 == p2
                    c1 = S.coef(S.analyse(I, a1, ())) * S.coef(S.analyse(I - {a1}, j, ((a2, m),)))
                    c2 = S.coef(S.analyse(I, j, ((a2, m),))) * S.coef(S.analyse((I - {j, m}) | {a2}, a1, ()))
                    assert c1 == -c2
                    st["C2 cases"] += 1
        # (C4)
        if a1 not in Ip and a2 not in Ip:
            for j in sorted(Ip):
                if not (a2 < j and S.adj(S.col(j), c)):
                    continue
                for jp in sorted(Ip):
                    if jp == j:
                        continue
                    four = a2 < jp and S.adj(S.col(jp), c)
                    if four:
                        continue        # Prop. 10.2, tested separately; d^2 = 0 is tested above
                    p1 = beta(I, j, ((a2, m),)) * beta((I - {j, m}) | {a2}, jp, ())
                    p2 = beta(I, jp, ()) * beta(I - {jp}, j, ((a2, m),))
                    assert p1 == p2
                    c1 = S.coef(S.analyse(I, j, ((a2, m),))) * S.coef(S.analyse((I - {j, m}) | {a2}, jp, ()))
                    c2 = S.coef(S.analyse(I, jp, ())) * S.coef(S.analyse(I - {jp}, j, ((a2, m),)))
                    assert c1 == -c2
                    st["C4 two-term cases"] += 1
    return dict(st)


def prop102(nwords, maxlen, seed):
    rng = random.Random(seed)
    st = defaultdict(int)
    done = 0
    while done < nwords:
        ct = rng.choice([["A", 2], ["A", 3], ["A", 4], ["D", 4]])
        rank = ct[1]
        L = rng.randint(4, maxlen)
        word = tuple(rng.randint(1, rank) for _ in range(L))
        red = rng.randint(1, rank)
        cases = []
        for a, j, jp, b in itertools.combinations(range(1, L + 1), 4):
            c, e, ep = word[a - 1], word[j - 1], word[jp - 1]
            if word[b - 1] != c:
                continue
            E = dynkin_edges(ct)
            if frozenset((c, e)) not in E or frozenset((c, ep)) not in E:
                continue
            if any(word[s - 1] == c for s in range(a + 1, b)):
                continue                                              # (H1)
            if any(word[s - 1] == e for s in range(1, j)) or any(word[s - 1] == ep for s in range(1, jp)):
                continue                                              # (H2)
            cases.append((a, j, jp, b))
        if not cases:
            continue
        done += 1
        S = Setup(ct, red, word)
        for a, j, jp, b in cases:
            c, e, ep = word[a - 1], word[j - 1], word[jp - 1]
            others = [s for s in range(1, L + 1) if s not in (a, j, jp, b)]
            for _ in range(3):
                I = frozenset(rng.sample(others, rng.randint(0, len(others)))) | {j, jp, b}
                K = (I - {j, jp, b}) | {a}
                src = S.arr(I)
                pos = {s: (s if s in I else -s) for s in range(1, L + 1)}
                pos["W"] = 0; pos[j] = -j; pos[jp] = -jp; pos[a] = -b; pos[b] = a
                tgt = sorted(pos, key=lambda lab: pos[lab])
                b0 = S.el(src, rand_reduced(src, tgt, rng))
                xA, xM, xj, xjp = S.dotvar(src, a), S.dotvar(src, b), S.dotvar(src, j), S.dotvar(src, jp)
                t = S.t

                def be(V, mov, F):
                    return S.el(S.arr(V), S.beta_word(V, mov, F, rng))
                pr = ((a, b),)
                P1 = be(I, j, ()) * be(I - {j}, jp, pr)
                P4 = be(I, jp, pr) * be((I - {jp, b}) | {a}, j, ())
                P2 = be(I, j, pr) * be((I - {j, b}) | {a}, jp, ())
                P3 = be(I, jp, ()) * be(I - {jp}, j, pr)
                assert P1 == (t(c, e) * xA + t(e, c) * xj) * b0, ("P1", ct, word, sorted(I))
                assert P4 == (t(c, e) * xM + t(e, c) * xj) * b0, ("P4", ct, word, sorted(I))
                assert P2 == (t(c, ep) * xM + t(ep, c) * xjp) * b0, ("P2", ct, word, sorted(I))
                assert P3 == (t(c, ep) * xA + t(ep, c) * xjp) * b0, ("P3", ct, word, sorted(I))
                # the only products from I to K, and the sum with the formula's coefficients
                found = []
                tot = None
                for t1 in S.my_terms(I):
                    for t2 in S.my_terms(t1.J):
                        if t2.J == K:
                            found.append(((t1.j, t1.F), (t2.j, t2.F)))
                            x = (S.coef(t1) * S.el(src, S.beta_word(I, t1.j, t1.F, rng))) * \
                                (S.coef(t2) * S.el(S.arr(t1.J), S.beta_word(t1.J, t2.j, t2.F, rng)))
                            tot = x if tot is None else tot + x
                assert sorted(found) == sorted([((j, ()), (jp, pr)), ((jp, pr), (j, ())), ((j, pr), (jp, ())), ((jp, ()), (j, pr))]), found
                assert tot.is_zero()
                st["10.2 cases"] += 1
                st["10.2 cases with a slot of colour c below a"] += any(word[s - 1] == c for s in range(1, a))
                st["10.2 cases with c = red colour or e/e' = red colour"] += (red in (c, e, ep))
    return st


if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1] == "C":
        ct = [sys.argv[2], int(sys.argv[3])]
        maxlen, nproc = int(sys.argv[4]), int(sys.argv[5])
        jobs = []
        for w in wordsC(ct[1], maxlen):
            for red in range(1, ct[1] + 1):
                jobs.append((ct, red, w, len(jobs)))
        tot = defaultdict(int)
        with Pool(nproc) as pool:
            for st in pool.imap_unordered(checkC, jobs, chunksize=2):
                for k_, v in st.items():
                    tot[k_] += v
        print("RESULT Theorem C %s%d maxlen %d: %d (word, red) cases, no assertion failed [%.0fs]" % (ct[0], ct[1], maxlen, len(jobs), time.time() - t0))
        for k_ in sorted(tot):
            print("   %-30s %d" % (k_, tot[k_]))
    else:
        st = prop102(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        print("RESULT Prop 10.2 [%.0fs]" % (time.time() - t0))
        for k_ in sorted(st):
            print("   %-60s %d" % (k_, st[k_]))
