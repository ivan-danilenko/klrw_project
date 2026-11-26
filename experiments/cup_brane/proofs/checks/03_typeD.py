"""
Type D (Theorem D, Lemmas 4.2, 4.3, Prop. 4.1, Cor. 4.4 of proofs/03_cyclic_combinatorics.md).  Pure Python.

    python3 proofs/checks/03_typeD.py [NMAX]          (default NMAX = 11)

Half-spin omega_n of D_n, model P_n = {(u, v) : 1 <= u < v <= n}, componentwise order,
colour(u, v) = n - (v - u) if v - u >= 2, and n (u odd) / n - 1 (u even) if v = u + 1.
 1. Model check (4 <= n <= NMAX): the row-major word of P_n is a chain omega_n -> lowest
    weight, and its heap order is the componentwise order; the order ideals are the strict
    partitions l_1 > l_2 > ... with l_u <= n - u (row u = {(u, u+1), ..., (u, u+l_u)}).
 2. For every strict partition: exhaustive search; compare with  l_3 >= 3 and l_4 >= 1;
    for the non-cyclic ones also the literal pattern detector and (n <= 9) the brute force
    over all configurations; check that non-cyclic ideals have colour multiplicities <= 3.
 3. Vector representation omega_1: chain 1, 2, ..., n-2, n-1, n, n-2, ..., 1 is a chain to the
    lowest weight; every colour occurs at most twice; no ideal is cyclic.
"""
import importlib, os, sys, itertools
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("03_lib")
Pt = importlib.import_module("03_patterns")
Mi = importlib.import_module("03_minuscule")


def colour(n, u, v):
    r = v - u
    if r >= 2:
        return n - r
    return n if u % 2 == 1 else n - 1


def elements(n):
    return [(u, v) for u in range(1, n) for v in range(u + 1, n + 1)]      # row-major


def strict_partitions(n):
    """l_1 > l_2 > ... > l_t > 0 with l_u <= n - u."""
    def rec(u, prev):
        yield ()
        for l in range(min(prev - 1, n - u), 0, -1):
            for rest in rec(u + 1, l):
                yield (l,) + rest
    return list(rec(1, n + 1))


def ideal_word(n, lam):
    E = [(u, v) for u in range(1, len(lam) + 1) for v in range(u + 1, u + lam[u - 1] + 1)]
    return E, [colour(n, u, v) for (u, v) in E]


def model_check(n):
    E = elements(n)
    word = [colour(n, u, v) for (u, v) in E]
    ws = L.chain_weights("D", n, n, word)
    assert all(ws[-1][i] != 1 for i in range(1, n + 1))
    less = L.heap_order(word, L.adjacency("D", n))
    for x, (u, v) in enumerate(E):
        for y, (u2, v2) in enumerate(E):
            assert (y in less[x]) == ((u, v) != (u2, v2) and u <= u2 and v <= v2), (n, (u, v), (u2, v2))
    ideals = L.order_ideals(len(word), less)
    assert len(ideals) == len(strict_partitions(n)) == 2 ** (n - 1)
    return word, less


def main(NMAX):
    for n in range(4, NMAX + 1):
        model_check(n)
        cnt = Counter()
        for lam in strict_partitions(n):
            E, col = ideal_word(n, lam)
            adj = L.adjacency("D", n)
            W = L.Word(col, adj)
            cyc = W.search() is not None
            exp = len(lam) >= 4 and lam[2] >= 3
            assert cyc == exp, (n, lam)
            if not exp:
                assert max(Counter(col).values(), default=0) <= 3, (n, lam)
                assert not Pt.patterns(col, adj)
                if n <= 9:
                    assert W.brute() is None
                    cnt["brute"] += 1
            else:
                ix = {e: i for i, e in enumerate(E)}
                j = ix[(3, 5)]
                F = tuple(sorted([(ix[(1, 3)], ix[(2, 4)]), (ix[(1, 4)], ix[(3, 6)]), (ix[(2, 3)], ix[(4, 5)])]))
                assert W.is_cyclic_config(j, F)
            cnt[cyc] += 1
        print("D%d half-spin: model ok; all %d strict partitions: cyclic <=> l_3 >= 3 and l_4 >= 1  (%d cyclic%s)" % (
            n, cnt[True] + cnt[False], cnt[True], ", brute force on the %d others" % cnt["brute"] if cnt["brute"] else ""))
        # vector representation
        vword = list(range(1, n - 1)) + [n - 1, n] + list(range(n - 2, 0, -1))
        ws = L.chain_weights("D", n, 1, vword)
        assert all(ws[-1][i] != 1 for i in range(1, n + 1))
        assert max(Counter(vword).values()) <= 2
        adj = L.adjacency("D", n)
        less = L.heap_order(vword, adj)
        for I in L.order_ideals(len(vword), less):
            assert L.Word([vword[s] for s in I], adj).search() is None
    print("D_n vector representation, 4 <= n <= %d: chain checked, multiplicities <= 2, no cyclic ideal" % NMAX)
    # mu_c = weight of I_(5,4,3,1): epsilon-coordinates and root content (Cor. 4.4(b))
    from fractions import Fraction as Fr
    for n in range(6, NMAX + 1):
        E, col = ideal_word(n, (5, 4, 3, 1))
        mu = L.chain_weights("D", n, n, col)[-1]
        eps = [Fr(0)] * (n + 1)
        for i in range(1, n + 1):                  # omega_i in epsilon-coordinates
            if i <= n - 2:
                for t in range(1, i + 1):
                    eps[t] += mu[i]
            else:
                for t in range(1, n + 1):
                    eps[t] += Fr(mu[i], 2) * (-1 if (i == n - 1 and t == n) else 1)
        want = [1] * (n - 6) + [-1, -1, -1, 1, -1, 1]
        assert [int(2 * e) for e in eps[1:]] == want, (n, eps)
        cont = Counter(col)
        assert [cont[c] for c in range(n - 5, n + 1)] == [1, 2, 3, 3, 2, 2] and len(col) == 13
    print("mu_c = 1/2(+,...,+,-,-,-,+,-,+), omega_n - mu_c = (1,2,3,3,2,2) on nodes n-5..n: n = 6..%d" % NMAX)
    for n in range(4, min(NMAX, 9) + 1):
        Mi.weight_order_check("D", n, n)
        Mi.weight_order_check("D", n, 1)
    print("weight order = reverse inclusion of ideals: D_n half-spin and vector, n <= %d" % min(NMAX, 9))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
