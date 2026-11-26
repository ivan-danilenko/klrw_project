"""
Referee check 0: (a) the relations with the signs used in 05_main_sliding.md (README conventions),
on every state with 2 or 3 strands over colours {1,2,3} of A_3 plus one red strand;
(b) my brute-force implementation of the tree formula (NOTES.md section 2) against formula.Stack.

    sage -python R05_basic.py
"""
import sys, itertools, random
from R05_lib import *


def relations():
    n_checks = 0
    cols = [1, 2, 3]
    for red in (1, 2):
        # two strands
        for a, b in itertools.product(cols + ["W"], repeat=2):
            if (a, b).count("W") > 1:
                continue
            word = [c for c in (a, b) if c != "W"]
            S = Setup(["A", 3], red, word)
            # arrangement labels: slots in order of appearance
            arr, k = [], 0
            for c in (a, b):
                if c == "W":
                    arr.append("W")
                else:
                    k += 1
                    arr.append(k)
            sq = S.el(arr, (1, 1))
            e = S.idem(arr)
            if "W" in (a, b):
                blk = arr[0] if a != "W" else arr[1]
                c = S.col(blk)
                exp = S.u(c) * S.dot(arr, blk) if c == red else e
            elif a == b:
                exp = 0 * e
                x1, x2, psi = S.dot(arr, arr[0]), S.dot(arr, arr[1]), S.el(arr, (1,))
                # dots are indexed by position at the state where they are written: the state is (c, c) throughout
                assert x1 * psi - psi * x2 == S.r(a) * e
                assert psi * x1 - x2 * psi == S.r(a) * e
                n_checks += 2
            elif S.adj(a, b):
                exp = S.t(a, b) * S.dot(arr, arr[0]) + S.t(b, a) * S.dot(arr, arr[1])
            else:
                exp = e
            assert sq == exp, (red, a, b, sq, exp)
            n_checks += 1
            # dots slide through crossings of different colours / red
            if a != b:
                psi = S.el(arr, (1,))
                arr2 = [arr[1], arr[0]]
                for lab in arr:
                    if lab != "W":
                        assert S.dot(arr, lab) * psi == psi * S.dot(arr2, lab)
                        n_checks += 1
        # three strands
        for a, b, c in itertools.product(cols + ["W"], repeat=3):
            if (a, b, c).count("W") > 1:
                continue
            word = [x for x in (a, b, c) if x != "W"]
            S = Setup(["A", 3], red, word)
            arr, k = [], 0
            for x in (a, b, c):
                if x == "W":
                    arr.append("W")
                else:
                    k += 1
                    arr.append(k)
            diff = S.el(arr, (1, 2, 1)) - S.el(arr, (2, 1, 2))
            e = S.idem(arr)
            if a == c and a != "W" and b != "W" and S.adj(a, b):
                exp = S.r(a) * S.t(a, b) * e
            elif a == c and a == red and b == "W":
                exp = S.r(a) * S.u(a) * e
            else:
                exp = 0 * e
            assert diff == exp, (red, a, b, c, diff, exp)
            n_checks += 1
    print("relations with signs: %d checks, all exact" % n_checks)


def formula_vs_stack(n_words, seed):
    rng = random.Random(seed)
    n_terms = 0
    for trial in range(n_words):
        ct = rng.choice([["A", 2], ["A", 3], ["A", 4], ["D", 4], ["D", 5]])
        rank = ct[1]
        L = rng.randint(2, 8)
        word = tuple(rng.randint(1, rank) for _ in range(L))
        S = Setup(ct, 1, word)
        st = S.stack
        for size in range(1, L + 1):
            for I in itertools.combinations(range(1, L + 1), size):
                I = frozenset(I)
                mine = {}
                for t in S.my_terms(I):
                    mono = defaultdict(int)
                    for cp, cx in t.mono:
                        mono["r_%d" % cp] += 1
                        mono["t_%d_%d" % (cp, cx)] += 1
                    sigma = list(range(1, L + 1))
                    for a, b in t.F:
                        sigma[a - 1], sigma[b - 1] = b, a
                    key = (t.J, tuple(sigma), tuple(sorted(mono.items())))
                    assert key not in mine
                    mine[key] = t.sign
                theirs = {(J, sigma, mono): sign for J, sigma, mono, sign in st.terms_from(I)}
                assert mine == theirs, (ct, word, sorted(I))
                n_terms += len(mine)
    print("tree formula: my brute-force implementation == formula.Stack on %d random words (arbitrary), %d terms" % (n_words, n_terms))


if __name__ == "__main__":
    relations()
    formula_vs_stack(int(sys.argv[1]) if len(sys.argv) > 1 else 60, 5)
