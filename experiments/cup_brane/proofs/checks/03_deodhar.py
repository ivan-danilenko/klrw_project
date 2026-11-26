"""
Section 7 of proofs/03_cyclic_combinatorics.md: cyclic versus Deodhar.  Pure Python.

    python3 proofs/checks/03_deodhar.py [--big]

Part 1 (type A, patterns; this is the computational shadow of Proposition 7.2).  For every
k-subset S of [n], n <= 12: the Grassmannian permutation w_S (w(1..k) = S increasing,
w(k+1..n) = complement increasing) is 321-avoiding; it contains one of the four hexagon
patterns of Billey-Warrington (7) iff the partition lambda(S), lambda_i = s_{k+1-i} - (k+1-i),
contains (4,4,4,3); and lambda(S) is the shape of the order ideal of the weight e_S
(the chain of e_S computed from the minuscule poset has the same set S).

Part 2 (Hecke algebra).  For every order ideal of the minuscule posets listed, test the
Deodhar property  C'_w = C'_{s_1} ... C'_{s_l}  (Billey-Warrington Thm 1 (4); Billey-Jones
Thm 2.3 (5)) directly in the Hecke algebra (Soergel normalisation; the product is the KL
basis element iff every coefficient of H_y, y != w, lies in v Z[v]), and compare with
cyclicity (exhaustive search).  --big adds D9 half-spin and A9 omega_5 (slow).
"""
import importlib, os, sys, itertools, time
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("03_lib")
M = importlib.import_module("03_minuscule")

# ----------------------------------------------------------------------------- Part 1

def contains_pattern(w, p):
    k = len(p)
    for pos in itertools.combinations(range(len(w)), k):
        vals = [w[i] for i in pos]
        order = sorted(range(k), key=lambda i: vals[i])
        rank = [0] * k
        for r, i in enumerate(order):
            rank[i] = r + 1
        if rank == list(p):
            return True
    return False


HEX = [(4, 6, 7, 1, 8, 2, 3, 5), (4, 6, 7, 8, 1, 2, 3, 5), (5, 6, 7, 1, 8, 2, 3, 4), (5, 6, 7, 8, 1, 2, 3, 4)]


def part1(NMAX=12):
    tested = 0
    for n in range(2, NMAX + 1):
        for k in range(1, n):
            for S in itertools.combinations(range(1, n + 1), k):
                T = [x for x in range(1, n + 1) if x not in S]
                w = list(S) + T
                lam = tuple(x for x in (S[k - i] - (k + 1 - i) for i in range(1, k + 1)) if x > 0)
                big = len(lam) >= 4 and lam[2] >= 4 and lam[3] >= 3
                assert not contains_pattern(w, (3, 2, 1))
                hexa = any(contains_pattern(w, p) for p in HEX) if n >= 8 else False
                assert hexa == big, (n, S, lam)
                # the two-descent patterns never occur in a Grassmannian permutation
                if n >= 8:
                    assert not contains_pattern(w, HEX[0]) and not contains_pattern(w, HEX[2])
                tested += 1
    print("Part 1: all %d Grassmannian permutations with n <= %d: 321-avoiding; hexagon pattern <=> lambda >= (4,4,4,3)" % (tested, NMAX))
    # the ideal of e_S has shape lambda(S)
    for n in range(3, 10):
        for k in range(1, n):
            word, adj, less, ideals, wt = M.setup("A", n - 1, k)
            for I in ideals:
                sub = [word[s] for s in I]
                # e_S: apply the chain to the set {1..k}: s_i swaps i and i+1
                S = set(range(1, k + 1))
                for i in sub:
                    assert i in S and i + 1 not in S       # a chain step moves i to i+1
                    S = (S - {i}) | {i + 1}
                S = sorted(S)
                lam = sorted((S[k - i] - (k + 1 - i) for i in range(1, k + 1)), reverse=True)
                # shape of the ideal = number of slots of each colour along diagonals:
                # colour k + d occurs  #{i : lambda_i - i >= d - ... }  -- compare contents instead
                cont = defaultdict(int)
                for r in range(1, k + 1):
                    for s in range(1, lam[r - 1] + 1):
                        cont[k + s - r] += 1
                cI = defaultdict(int)
                for c in sub:
                    cI[c] += 1
                assert dict(cont) == dict(cI), (n, k, S)
    print("Part 1: for sl_3 .. sl_9 the ideal of e_S has the colour content of lambda(S)")

# ----------------------------------------------------------------------------- Part 2

def is_deodhar(A, n, word):
    """Product C'_{s_1}...C'_{s_l} over the Weyl group of Cartan matrix A (1-based), group
    elements x encoded by x^{-1}(rho) in fundamental-weight coordinates."""
    def refl(mu, s):
        c = mu[s]
        return tuple(mu[j] - c * A[s][j] for j in range(n + 1))
    rho = tuple([0] + [1] * n)
    elt = {rho: {0: 1}}
    for s in word:
        new = defaultdict(lambda: defaultdict(int))
        for mu, p in elt.items():
            nu = refl(mu, s)
            up = mu[s] > 0                      # x s > x
            for e, c in p.items():
                new[nu][e] += c
                new[mu][e + (1 if up else -1)] += c
        elt = {}
        for mu, p in new.items():
            q = {e: c for e, c in p.items() if c}
            if q:
                elt[mu] = q
    w = rho
    for s in word:
        w = refl(w, s)
    for mu, p in elt.items():
        if mu == w:
            assert p == {0: 1}
        elif min(p) <= 0:
            return False
    return True


def part2(cases):
    for typ, n, k in cases:
        t0 = time.time()
        word, adj, less, ideals, wt = M.setup(typ, n, k)
        A = L.cartan(typ, n)
        nd = cy = agree = 0
        mism = []
        for I in ideals:
            sub = [word[s] for s in I]
            d = is_deodhar(A, n, sub)
            c = L.Word(sub, adj).search() is not None
            nd += not d
            cy += c
            if d == (not c):
                agree += 1
            else:
                mism.append((len(I), "non-Deodhar" if not d else "Deodhar", "cyclic" if c else "not cyclic"))
        print("Part 2: %s%d omega_%d: %d ideals, %d non-Deodhar, %d cyclic; %s  [%.0fs]" % (
            typ, n, k, len(ideals), nd, cy,
            "non-Deodhar <=> cyclic on every ideal" if not mism else "MISMATCHES (size, Deodhar?, cyclic?): %s" % sorted(mism),
            time.time() - t0), flush=True)


if __name__ == "__main__":
    part1()
    cases = [("A", 7, 4), ("A", 8, 4), ("A", 8, 3), ("D", 6, 6), ("D", 7, 7), ("D", 8, 8), ("D", 6, 1),
             ("D", 7, 1), ("E", 6, 1), ("E", 6, 6), ("E", 7, 7)]
    if "--big" in sys.argv:
        cases += [("A", 9, 5), ("D", 9, 9)]
    part2(cases)
