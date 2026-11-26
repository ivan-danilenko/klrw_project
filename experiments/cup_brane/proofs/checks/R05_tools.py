"""
Referee check 2a: the general tools of sections 1, 2, 4 of 05_main_sliding.md, tested from their
statements on random instances (arbitrary states, red strand anywhere, non-reduced words).

    sage -python R05_tools.py NTRIALS SEED

* Lemma 1.2 (Straightening): if (S1), (S2) then psi_D = psi_D' for random reduced D'.
  Also counts how often the conclusion fails when (S1) or (S2) fails (hypotheses not vacuous).
* Lemma 1.3 (bigons), Lemma 1.4 (vanishing).
* Lemma 2.2 (Sliding Lemma) for random words, with the error words E_t and the constants gamma_t.
* Lemma 4.2: mu mu^* = P e_0 and mu^* mu = P e_1.
"""
import sys, random, itertools
from R05_lib import *

TYPES = [["A", 2], ["A", 3], ["A", 4], ["D", 4]]


def random_setup(rng, nmin=3, nmax=6):
    ct = rng.choice(TYPES)
    rank = ct[1]
    n = rng.randint(nmin, nmax)
    # few colours so that equal colours are frequent
    cols = rng.sample(range(1, rank + 1), min(rank, rng.randint(2, 3)))
    word = tuple(rng.choice(cols) for _ in range(n))
    red = rng.choice(cols + list(range(1, rank + 1)))
    return ct, red, word


def straighten(ntrials, rng):
    st = defaultdict(int)
    for _ in range(ntrials):
        ct, red, word = random_setup(rng)
        S = Setup(ct, red, word)
        arr = list(range(1, len(word) + 1)) + ["W"]
        rng.shuffle(arr)
        L = rng.randint(2, 11)
        D = tuple(rng.randint(1, len(arr) - 1) for _ in range(L))
        n, _ = crossing_counts(arr, D)
        target = apply_word(arr, D)
        reduced = all(v <= 1 for v in n.values())
        s1, s2 = check_S1_S2(S, arr, D)
        x = S.el(arr, D)
        y = S.el(arr, rand_reduced(arr, target, rng))
        if s1 and s2:
            st["1.2 cases"] += 1
            st["1.2 non-reduced"] += (not reduced)
            if x != y:
                st["1.2 FAIL"] += 1
                print("FAIL 1.2", ct, red, word, arr, D)
        else:
            st["hyp. violated"] += 1
            st["hyp. violated and psi_D != psi_D'"] += (x != y)
        if not s2:
            continue
        # classify the pairs crossing at least twice
        kinds = defaultdict(list)
        for pair, cnt in n.items():
            if cnt < 2:
                continue
            a, b = tuple(pair)
            ca, cb = S.col(a), S.col(b)
            if "W" in (ca, cb):
                blk = a if cb == "W" else b
                kinds["adj" if S.col(blk) == red else "free"].append((blk, "W", cnt))
            elif ca == cb:
                kinds["same"].append((a, b, cnt))
            elif S.adj(ca, cb):
                kinds["adj"].append((a, b, cnt))
            else:
                kinds["free"].append((a, b, cnt))
        # Lemma 1.4
        if kinds["same"] and not kinds["adj"]:
            st["1.4 cases"] += 1
            if not x.is_zero():
                st["1.4 FAIL"] += 1
                print("FAIL 1.4", ct, red, word, arr, D)
        # Lemma 1.3
        if not kinds["same"]:
            ok = True
            for a, b, cnt in kinds["adj"]:
                for z in (a, b):
                    if z == "W":
                        continue
                    for z2 in arr:
                        if z2 != z and z2 != "W" and S.col(z2) == S.col(z) and n[frozenset((z, z2))] > 0:
                            ok = False
            if ok and kinds["adj"]:
                Q = S.idem(arr)
                for a, b, cnt in kinds["adj"]:
                    if b == "W":
                        q = S.u(red) * S.dot(arr, a)
                    else:
                        q = S.t(S.col(a), S.col(b)) * S.dot(arr, a) + S.t(S.col(b), S.col(a)) * S.dot(arr, b)
                    for _ in range(cnt // 2):
                        Q = Q * q
                st["1.3 cases"] += 1
                if x != Q * y:
                    st["1.3 FAIL"] += 1
                    print("FAIL 1.3", ct, red, word, arr, D)
    return st


def gamma(S, c, left, right):
    """Constant of Lemma 2.1 for a letter with left strand `left`, right strand `right`; M has colour c."""
    cl, cr = S.col(left), S.col(right)
    if cl == "W" or cl != c:
        return None
    if cr == "W":
        return S.r(c) * S.u(c) if c == S.red else None
    if S.adj(c, cr):
        return S.r(c) * S.t(c, cr)
    return None


def sliding(ntrials, rng):
    st = defaultdict(int)
    for _ in range(ntrials):
        ct, red, word = random_setup(rng, 2, 5)
        c = rng.choice(list(set(word)) + [rng.randint(1, ct[1])])
        n = len(word)
        S = Setup(ct, red, word + (c,))           # slot n+1 is the strand M
        M = n + 1
        s = list(range(1, n + 1)) + ["W"]
        rng.shuffle(s)
        N = len(s)                                  # number of strands of s
        L = rng.randint(0, 9)
        beta = tuple(rng.randint(1, N - 1) for _ in range(L))
        s2 = apply_word(s, beta)
        X_s = S.el(s + [M], tuple(range(N, 0, -1)))
        X_s2 = S.el(s2 + [M], tuple(range(N, 0, -1)))
        lhs = S.el(s + [M], beta) * X_s2 - X_s * S.el([M] + s, tuple(i + 1 for i in beta))
        _, letters = crossing_counts(s, beta)
        rhs = 0 * lhs
        for t, (left, right) in enumerate(letters):
            g = gamma(S, c, left, right)
            if g is None:
                continue
            st["bad letters"] += 1
            it = beta[t]
            E = beta[:t] + tuple(range(N, it + 1, -1)) + tuple(range(it - 1, 0, -1)) + tuple(i + 1 for i in beta[t + 1:])
            rhs = rhs + g * S.el(s + [M], E)
        st["2.2 cases"] += 1
        if lhs != rhs:
            st["2.2 FAIL"] += 1
            print("FAIL 2.2", ct, red, word, c, s, beta)
    return st


def lemma42(ntrials, rng):
    st = defaultdict(int)
    for _ in range(ntrials):
        ct = rng.choice(TYPES)
        rank = ct[1]
        n = rng.randint(2, 6)
        word = tuple(rng.randint(1, rank) for _ in range(n))
        red = rng.randint(1, rank)
        c = rng.randint(1, rank)
        S = Setup(ct, red, word + (c,))
        m = n + 1
        a = rng.randint(1, n)
        first = word.index(word[a - 1]) == a - 1
        Kpp = set(rng.sample(range(1, n + 1), rng.randint(1, n))) | {a}
        for side in ("L", "R"):
            # iota_L: vertex K'' of w (m on the left); iota_R: vertex K'' + m
            V = Kpp if side == "L" else Kpp | {m}
            e0 = S.arr(V)
            e1 = S.arr(V - {a})
            p = e0.index(a) + 1
            crossed = [lab for lab in e0 if lab != a and (lab == "W" or lab < a)]
            l = len(crossed)
            mu = tuple(range(p - 1, p - l - 1, -1))
            assert apply_word(e0, mu) == e1
            mus = tuple(reversed(mu))
            P0 = S.idem(e0); P1 = S.idem(e1)
            e = S.col(a)
            for y in crossed:
                if y == "W":
                    if e == red:
                        P0 = P0 * (S.u(e) * S.dot(e0, a)); P1 = P1 * (S.u(e) * S.dot(e1, a))
                elif S.adj(e, S.col(y)):
                    P0 = P0 * (S.t(e, S.col(y)) * S.dot(e0, a) + S.t(S.col(y), e) * S.dot(e0, y))
                    P1 = P1 * (S.t(e, S.col(y)) * S.dot(e1, a) + S.t(S.col(y), e) * S.dot(e1, y))
            x0 = S.el(e0, mu + mus); x1 = S.el(e1, mus + mu)
            if first:
                st["4.2 cases (first occurrence)"] += 1
                if x0 != P0 or x1 != P1 or P0.is_zero():
                    st["4.2 FAIL"] += 1
                    print("FAIL 4.2", ct, red, word, c, a, sorted(Kpp), side)
            else:
                st["not first occurrence"] += 1
                st["not first: mu mu^* != P"] += (x0 != P0)
    return st


if __name__ == "__main__":
    ntrials = int(sys.argv[1]); seed = int(sys.argv[2])
    rng = random.Random(seed)
    for name, f, k in (("straighten", straighten, 1), ("sliding", sliding, 1), ("lemma42", lemma42, 1)):
        st = f(ntrials // k, rng)
        print(name, dict(sorted(st.items())), flush=True)
