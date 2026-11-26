"""
Referee check 2b: Lemma 3.2 / Prop. 3.3 and every lemma of section 8 of 05_main_sliding.md, tested
from the statements in the text, with my own formula implementation and random reduced words.

    sage -python R05_sec8.py word  TYPE RANK RED WORD [SEED]
    sage -python R05_sec8.py all   TYPE RANK MINLEN MAXLEN NPROC [reds: all|one]
    sage -python R05_sec8.py sample TYPE RANK NWORDS LEN SEED NPROC

Only words w = w'.c in which every colour occurs at most twice AND the last colour occurs twice
are used (the case treated in section 8).  For such a word, with a = the twin of m:

 3.2  the terms of w at I in terms of those of w' (kinds 1, alpha, beta, gamma; signs)
 8.1  mover a: exact slide
 8.3  frozen term: number of bad letters, its strands, gamma
 8.4  data U, families (t_1(S), t_2(S)), bijection onto all pairs (t_1, t_2)
 8.5  D(S) satisfies (S1), (S2), has permutation sigma_U, and beta(t_1(S)) beta(t_2(S)) = B
 8.6  E_t satisfies (S1), (S2), permutation sigma_U, psi_{E_t} = B(j, F', {}), and the slide formula
 8.7  coef(t_1(S)) coef(t_2(S)) = (-1)^{k-|S|} C;  for T = {}: = coef'(j,F') r_c t_{c,c(N)}
 E    E_{I',K'} = 0 for ALL I', K' (computed from the definition in section 3), and the
      cocycle identity of Lemma 4.1
"""
import sys, random, itertools, time
from multiprocessing import Pool
from R05_lib import *
from R05_tools import gamma


def sigmaU_target(S, I, j, pairs, a, m):
    """Final arrangement of sigma_U (section 8.3) for strands labelled by their slots at I = I' + m."""
    pos = {s: (s if s in I else -s) for s in range(1, S.m + 1)}
    pos["W"] = 0
    pos[j] = -j
    pos[a] = -m
    pos[m] = -a
    for (x, y) in pairs:
        pos[x] = -y
        pos[y] = +x
    return sorted(pos, key=lambda lab: pos[lab])


def final_slots(tgt, pairs, a, m):
    """Relabel the strands of an arrangement (labelled by their slots at I) by the slots they occupy at the end."""
    lab = {a: m, m: a}
    for x, y in pairs:
        lab[x] = y; lab[y] = x
    return [lab.get(z, z) for z in tgt]


def check_word(args):
    ct, red, word, seed = args
    rng = random.Random(seed)
    st = defaultdict(int)
    word = tuple(word)
    m = len(word); n = m - 1
    c = word[-1]
    assert word[:-1].count(c) == 1 and all(word.count(x) <= 2 for x in set(word))
    a = word.index(c) + 1
    S = Setup(ct, red, word)
    Sp = Setup(ct, red, word[:-1])            # only used for the terms of w'
    Xword = tuple(range(m, 0, -1))            # X_s for a state s with m strands (n black + red)
    subsets = [frozenset(x) for k in range(n + 1) for x in itertools.combinations(range(1, n + 1), k)]

    def arrR(Ip):   # e_{I'}.c  = vertex I' + m of w
        return S.arr(set(Ip) | {m})

    def arrL(Ip):   # c.e_{I'}  = vertex I' of w
        return S.arr(Ip)

    terms_p = {Ip: Sp.my_terms(Ip) for Ip in subsets}
    # ---------------- Lemma 3.2 ----------------------------------------------------------------
    key = lambda t: (t.j, t.F)
    gam = {}
    for Ip in subsets:
        low = {key(t): t for t in S.my_terms(Ip)}
        prim = {key(t): t for t in terms_p[Ip]}
        assert set(low) == set(prim)
        for k_ in low:
            assert (low[k_].J, low[k_].sign, low[k_].mono) == (prim[k_].J, prim[k_].sign, prim[k_].mono)
        up = S.my_terms(Ip | {m})
        alpha = {key(t): t for t in up if t.j != m and all(m not in p for p in t.F)}
        beta = [t for t in up if t.j == m]
        gam[Ip] = [t for t in up if t.j != m and any(m in p for p in t.F)]
        assert set(alpha) == set(prim)
        for k_ in alpha:
            assert alpha[k_].J == prim[k_].J | {m} and alpha[k_].sign == -prim[k_].sign and alpha[k_].mono == prim[k_].mono
        assert len(beta) == 1 and beta[0].F == () and beta[0].sign == 1 and beta[0].J == Ip
        for t in gam[Ip]:
            assert (a, m) in t.F and a not in Ip and a in t.J and m not in t.J
        st["3.2 vertices"] += 1

    # elements ----------------------------------------------------------------------------------
    def X(Ip):
        return S.el(arrR(Ip), Xword)

    # d' embedded: for a term t' of w' at I', a random reduced word read in e_{I'} (m strands)
    def word_p(t):
        return rand_reduced(Sp.arr(t.I), Sp.target_arr(t.I, t.j, t.F), rng)

    dR = defaultdict(lambda: 0); dL = defaultdict(lambda: 0); Fm = defaultdict(lambda: 0)
    for Ip in subsets:
        for t in terms_p[Ip]:
            w_ = word_p(t)
            cf = S.coef(t)          # same parameters; S.coef only uses sign and monomial
            dR[Ip, t.J] = dR[Ip, t.J] + cf * S.el(arrR(Ip), w_)
            dL[Ip, t.J] = dL[Ip, t.J] + cf * S.el(arrL(Ip), tuple(i + 1 for i in w_))
        Fm[Ip, Ip] = Fm[Ip, Ip] + X(Ip)
        for t in gam[Ip]:
            Fm[Ip, t.J] = Fm[Ip, t.J] + S.coef(t) * S.el(arrR(Ip), S.beta_word(Ip | {m}, t.j, t.F, rng))
    Edef = {}
    for Ip in subsets:
        for Kp in subsets:
            tot = None
            for Jp in subsets:
                if (Ip, Jp) in dR and (Jp, Kp) in Fm:
                    x = dR[Ip, Jp] * Fm[Jp, Kp]
                    tot = x if tot is None else tot + x
                if (Ip, Jp) in Fm and (Jp, Kp) in dL:
                    x = Fm[Ip, Jp] * dL[Jp, Kp]
                    tot = -x if tot is None else tot - x
            if tot is not None:
                st["E components computed"] += 1
                st["E components with a notin K'"] += (a not in Kp)
                if not tot.is_zero():
                    st["E FAIL"] += 1
                    print("FAIL E", ct, red, word, sorted(Ip), sorted(Kp), flush=True)

    # ---------------- section 8 -------------------------------------------------------------------
    for Ip in subsets:
        I = Ip | {m}
        for t in terms_p[Ip]:
            if a in Ip and t.j != a:
                continue
            beta = word_p(t)
            sR = arrR(Ip)
            lhs = S.el(sR, beta) * X(t.J) - X(Ip) * S.el(arrL(Ip), tuple(i + 1 for i in beta))
            _, letters = crossing_counts(Sp.arr(Ip), beta)
            bad = [(i, l, r_) for i, (l, r_) in enumerate(letters) if gamma(S, c, l, r_) is not None]
            if a in Ip:
                # Lemma 8.1
                st["8.1 cases"] += 1
                assert not bad and lhs.is_zero(), ("8.1", word, sorted(Ip), t)
                continue
            # frozen term (a notin I', so a is unaffected); Lemma 8.3
            assert all(a not in p for p in t.F) and t.j != a
            nodes = [("mover", S.col(t.j))] + [(p, S.col(p[0])) for p in t.F]
            adjn = [x for x, cx in nodes if S.adj(cx, c)]
            assert len(adjn) <= 1                                   # Lemma 5.4
            assert all(cx != c for x, cx in nodes)
            N = adjn[0] if adjn else None
            inside = N is not None and ((N == "mover" and a < t.j) or (N != "mover" and N[0] < a < N[1]))
            st["8.3 frozen terms"] += 1
            assert len(bad) == (1 if inside else 0), ("8.3", word, sorted(Ip), t, bad)
            if not inside:
                assert lhs.is_zero()
                continue
            st["8.3 with a bad letter"] += 1
            i_t, left, right = bad[0]
            assert left == a and right == (t.j if N == "mover" else N[0])   # S' = mover or L_p (label a_p)
            cN = S.col(t.j) if N == "mover" else S.col(N[0])
            g = S.r(c) * S.t(c, cN)
            # Lemma 8.6
            it = beta[i_t]
            Et = beta[:i_t] + tuple(range(m, it + 1, -1)) + tuple(range(it - 1, 0, -1)) + tuple(i + 1 for i in beta[i_t + 1:])
            s1, s2 = check_S1_S2(S, sR, Et)
            tgt = sigmaU_target(S, I, t.j, t.F, a, m)
            assert s1 and s2 and apply_word(sR, Et) == tgt, ("8.6 hyp", word, sorted(Ip), t)
            assert final_slots(tgt, t.F, a, m) == S.arr(t.J)             # ends at K' (as a vertex of w)
            B = S.el(sR, rand_reduced(sR, tgt, rng))
            assert S.el(sR, Et) == B, ("8.6 B", word, sorted(Ip), t)
            assert lhs == g * B, ("8.6 formula", word, sorted(Ip), t)
            st["8.6 cases"] += 1
            # Lemma 8.7, case T = {}:  coef(t_1) coef(t_2) = coef'(j,F') r_c t_{c,c(N)}
            U = S.analyse(I, t.j, t.F + ((a, m),))
            assert U is not None and U.J == t.J | {a}
            t2 = S.analyse(U.J, a, ())
            assert S.coef(U) * S.coef(t2) == S.coef(t) * g, ("8.7 T=0", word, sorted(Ip), t)
            assert U.parent[(a, m)][0] == N                               # parent of (a,m) is N
            st["8.7 (T empty) cases"] += 1

        if a in Ip:
            continue
        # ---- data, Lemma 8.4, 8.5, 8.7 ----
        pairs_direct = set()
        for t1 in gam[Ip]:
            for t2 in S.my_terms(t1.J):
                if t2.j == a:
                    pairs_direct.add((t1.j, t1.F, t2.F))
                else:
                    # (8.1): the second factor of a contributing product has mover a if a notin K'
                    assert a in t2.J
        pairs_family = set()
        for U in gam[Ip]:
            # components of the link graph minus (a, m): recompute links from the definition
            nodes = ["mover"] + [p for p in U.F if p != (a, m)]
            def linked(x, y):
                if x == "mover" or y == "mover":
                    p = y if x == "mover" else x
                    return p[0] < U.j < p[1] and S.adj(S.col(p[0]), S.col(U.j))
                (a1, b1), (a2, b2) = x, y
                return ((a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1)) and S.adj(S.col(a1), S.col(a2))
            comp = {}
            for x in nodes:
                if x in comp:
                    continue
                comp[x] = x
                todo = [x]
                while todo:
                    y = todo.pop()
                    for z in nodes:
                        if z not in comp and linked(y, z):
                            comp[z] = x
                            todo.append(z)
            Fp = tuple(p for p in nodes if p != "mover" and comp[p] == "mover")
            roots = sorted(set(comp[p] for p in nodes if comp[p] != "mover"))
            T = [tuple(p for p in nodes if p != "mover" and comp[p] == r_) for r_ in roots]
            k = len(T)
            st["8.4 data with k=%d" % k] += 1
            # 8.4.1
            tp = Sp.analyse(Ip, U.j, Fp)
            assert tp is not None
            for Ti in T:
                q = [p for p in Ti if p[0] < a < p[1] and S.adj(S.col(p[0]), c)]
                assert len(q) == 1
            # (L1): no link between T and mover / F', nor between different T_i
            for i_, Ti in enumerate(T):
                for p in Ti:
                    assert not linked("mover", p) and not any(linked(p, q_) for q_ in Fp)
                    for Tj in T[i_ + 1:]:
                        assert not any(linked(p, q_) for q_ in Tj)
            allpairs = Fp + tuple(p for Ti in T for p in Ti)
            sR = S.arr(I)
            tgt = sigmaU_target(S, I, U.j, allpairs, a, m)
            B = S.el(sR, rand_reduced(sR, tgt, rng))
            Cref = None
            for r_ in range(k + 1):
                for Sset in itertools.combinations(range(k), r_):
                    TS = tuple(p for i_ in Sset for p in T[i_])
                    TSc = tuple(p for i_ in range(k) if i_ not in Sset for p in T[i_])
                    t1 = S.analyse(I, U.j, Fp + ((a, m),) + TS)
                    assert t1 is not None, "t_1(S) not admissible"
                    t2 = S.analyse(t1.J, a, TSc)
                    assert t2 is not None, "t_2(S) not admissible"
                    assert t2.J == U.J - {a}                         # K'_U
                    assert S.arr(t2.J) == final_slots(tgt, allpairs, a, m)
                    pairs_family.add((t1.j, t1.F, t2.F))
                    # Lemma 8.5
                    w1 = S.beta_word(I, t1.j, t1.F, rng)
                    w2 = S.beta_word(t1.J, t2.j, t2.F, rng)
                    D = w1 + w2
                    s1, s2 = check_S1_S2(S, sR, D)
                    assert s1 and s2 and apply_word(sR, D) == tgt, ("8.5 hyp", word, sorted(Ip), U, Sset)
                    ncnt, _ = crossing_counts(sR, D)
                    st["8.5 products with %d double crossings" % sum(1 for v in ncnt.values() if v == 2)] += 1
                    prod = S.el(sR, w1) * S.el(S.arr(t1.J), w2)
                    assert prod == B, ("8.5 B", word, sorted(Ip), U, Sset)
                    # Lemma 8.7
                    cc = S.coef(t1) * S.coef(t2) * ((-1) ** (k - len(Sset)))
                    if Cref is None:
                        Cref = cc
                    assert cc == Cref, ("8.7", word, sorted(Ip), U, Sset)
        assert pairs_direct == pairs_family, ("8.4.3 bijection", word, sorted(Ip))
        st["8.4 pairs (t1,t2)"] += len(pairs_direct)
    return ct, red, word, dict(st)


def T1_words_last_twice(rank, minlen, maxlen):
    for L in range(minlen, maxlen + 1):
        for w in itertools.product(range(1, rank + 1), repeat=L):
            if all(w.count(x) <= 2 for x in set(w)) and w.count(w[-1]) == 2:
                yield w


if __name__ == "__main__":
    mode = sys.argv[1]
    ct = [sys.argv[2], int(sys.argv[3])]
    rank = ct[1]
    t0 = time.time()
    if mode == "word":
        red = int(sys.argv[4]); word = tuple(int(x) for x in sys.argv[5])
        seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        print(check_word((ct, red, word, seed)))
        sys.exit()
    jobs = []
    if mode == "all":
        minlen, maxlen, nproc = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        reds = sys.argv[7] if len(sys.argv) > 7 else "all"
        for w in T1_words_last_twice(rank, minlen, maxlen):
            for red in (range(1, rank + 1) if reds == "all" else [1 + (sum(w) + len(jobs)) % rank]):
                jobs.append((ct, red, w, len(jobs)))
    else:
        nwords, L, seed, nproc = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        rng = random.Random(seed)
        while len(jobs) < nwords:
            w = rng.sample([x for x in range(1, rank + 1) for _ in range(2)], L)
            if w.count(w[-1]) != 2:
                continue
            jobs.append((ct, rng.randint(1, rank), tuple(w), len(jobs) + 1000 * seed))
    tot = defaultdict(int)
    nw = 0
    with Pool(nproc) as pool:
        for ct_, red, word, st in pool.imap_unordered(check_word, jobs, chunksize=2):
            nw += 1
            for k_, v in st.items():
                tot[k_] += v
            if nw % 200 == 0:
                print("  ... %d / %d [%.0fs]" % (nw, len(jobs), time.time() - t0), flush=True)
    print("RESULT %s %s%d: %d (word, red) cases, no assertion failed [%.0fs]" % (mode, ct[0], rank, nw, time.time() - t0))
    for k_ in sorted(tot):
        print("   %-45s %d" % (k_, tot[k_]))
