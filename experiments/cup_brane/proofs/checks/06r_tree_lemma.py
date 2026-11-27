"""
Revision of 06_main_intertwiner.md (point D6 of R06_referee.md): exact check of the tree
lemma, Lemma 6.7.

For a finite rooted tree T (root 0, parent function par), independent variables A_v, B_v
(v != 0) and M_v = A_v + B_v:

    sum over connected S containing the root of
        prod_{v in S, v != 0} B_v * prod_{v not in S, par(v) in S} A_v * prod_{v not in S, par(v) not in S} M_v
    =  prod_{v != 0} M_v .

Checked as an identity of polynomials over ZZ (sage) for EVERY rooted tree with at most
N vertices, up to relabelling: every rooted tree is isomorphic to a recursive tree
(par(v) < v), and all (n-1)! recursive trees with n <= N vertices are enumerated.

    cd proofs/checks && sage -python 06r_tree_lemma.py [N]        (default N = 7)
"""
import sys, itertools
from sage.all import PolynomialRing, ZZ


def recursive_trees(n):
    for par in itertools.product(*[range(v) for v in range(1, n)]):
        yield (None,) + par


def connected_root_sets(n, par):
    """all S containing 0 with par(v) in S for every v in S, v != 0."""
    for bits in itertools.product((0, 1), repeat=n - 1):
        S = {0} | {v for v in range(1, n) if bits[v - 1]}
        if all(par[v] in S for v in S if v):
            yield S


def check(N):
    total = 0
    for n in range(1, N + 1):
        if n == 1:
            total += 1                      # empty products: 1 = 1
            continue
        names = ["A%d" % v for v in range(1, n)] + ["B%d" % v for v in range(1, n)]
        P = PolynomialRing(ZZ, names)
        A = dict(zip(range(1, n), P.gens()[:n - 1]))
        B = dict(zip(range(1, n), P.gens()[n - 1:]))
        M = {v: A[v] + B[v] for v in A}
        rhs = P.one()
        for v in M:
            rhs *= M[v]
        cnt = 0
        for par in recursive_trees(n):
            lhs = P.zero()
            for S in connected_root_sets(n, par):
                t = P.one()
                for v in range(1, n):
                    if v in S:
                        t *= B[v]
                    elif par[v] in S:
                        t *= A[v]
                    else:
                        t *= M[v]
                lhs += t
            assert lhs == rhs, (n, par)
            cnt += 1
        total += cnt
        print("n = %d vertices: %d recursive trees, identity holds exactly" % (n, cnt), flush=True)
    return total


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    print("Lemma 6.7 checked on %d rooted trees (all shapes with <= %d vertices): OK" % (check(N), N))
