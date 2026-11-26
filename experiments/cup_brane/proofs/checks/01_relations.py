"""
01_relations.py   (sage -python proofs/checks/01_relations.py)

Checks, in the klrw implementation (all three scalings on, base ring Z, polynomial
parameters), every defining relation of the presentation of Definition A.1 of
01_foundations.md, with exact signs, on EVERY colour pattern of 2 and 3 strands built from

    black colours 1, 2, 3 of A_3   (1 ~ 2, 2 ~ 3, 1 and 3 not adjacent)
    one red strand W1 or W2        (red colour 1: adjacent to black 1 only "through u_1";
                                    red colour 2 likewise for black 2)

i.e. all sequences of length 2, 3 over {1, 2, 3, Wk} with at most one red strand, for
k = 1, 2.  Also the far-commutation relations on a sample of 4-strand states.

Relations checked (product a*b = first a, then b; positions 1-based, read in the state on the
side where the generator is written):
  (Q)  psi_1 psi_1 = Q_{s1 s2}(x_1, x_2)  on the state (s1, s2)
  (D)  x_1 psi - psi x_2,  psi x_1 - x_2 psi,  x_2 psi - psi x_1,  psi x_2 - x_1 psi
  (B)  psi_1 psi_2 psi_1 - psi_2 psi_1 psi_2  on (s1, s2, s3)
  (C)  psi_1 psi_3 = psi_3 psi_1;  x_i psi_j = psi_j x_i for i not in {j, j+1}
"""
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alg import Alg   # noqa: E402

ADJ = {frozenset((1, 2)), frozenset((2, 3))}
CT = ["A", 3]


def is_red(a):
    return isinstance(a, str)


def red_col(a):
    return int(a[1:])


def tok(a):
    return a if is_red(a) else str(a)


def seqstr(s):
    return " ".join(tok(a) for a in s)


def expected_Q(A, s):
    """Q_{s1 s2}(x_1, x_2) e(s) according to Definition A.1."""
    a, b = s
    e = A.e(seqstr(s))
    if not is_red(a) and not is_red(b):
        if a == b:
            return 0 * e
        if frozenset((a, b)) in ADJ:
            return A.t(a, b) * A.dot(seqstr(s), 1) + A.t(b, a) * A.dot(seqstr(s), 2)
        return e
    if is_red(b):
        return A.u(a) * A.dot(seqstr(s), 1) if a == red_col(b) else e
    return A.u(b) * A.dot(seqstr(s), 2) if b == red_col(a) else e


def expected_cubic(A, s):
    a, b, c = s
    e = A.e(seqstr(s))
    if not is_red(a) and not is_red(c) and a == c:
        if not is_red(b) and frozenset((a, b)) in ADJ:
            return A.r(a) * A.t(a, b) * e
        if is_red(b) and red_col(b) == a:
            return A.r(a) * A.u(a) * e
    return 0 * e


def act(s, i):
    s = list(s)
    s[i - 1], s[i] = s[i], s[i - 1]
    return tuple(s)


def main():
    blacks = [1, 2, 3]
    n_checks = 0
    fails = []
    for k in (1, 2):
        alphabet = blacks + ["W%d" % k]
        for L in (2, 3):
            for s in itertools.product(alphabet, repeat=L):
                if sum(is_red(a) for a in s) > 1:
                    continue
                A = Alg(CT, seqstr(s))
                S = seqstr(s)
                if L == 2:
                    # (Q)
                    got = A.word(S, (1, 1))
                    exp = expected_Q(A, s)
                    n_checks += 1
                    if got != exp:
                        fails.append(("Q", s, got, exp))
                    # (D)
                    t = act(s, 1)
                    T = seqstr(t)
                    psi = A.x(S, 1)
                    same = (not is_red(s[0]) and s[0] == s[1])
                    rr = A.r(s[0]) * A.e(S) if same else 0 * A.e(S)
                    if not is_red(s[0]):
                        # dot on the strand starting at position 1: before = x_1 on s, after = x_2 on t
                        d1 = A.dot(S, 1) * psi - psi * A.dot(T, 2)
                        n_checks += 1
                        if d1 != rr:
                            fails.append(("x1 psi - psi x2", s, d1, rr))
                    if not is_red(s[1]):
                        # dot on the strand starting at position 2: after = x_1 on t, before = x_2 on s
                        d2 = psi * A.dot(T, 1) - A.dot(S, 2) * psi
                        n_checks += 1
                        if d2 != rr:
                            fails.append(("psi x1 - x2 psi", s, d2, rr))
                    if not is_red(s[0]) and not is_red(s[1]):
                        # dots that do NOT follow their strand (only meaningful for equal colours)
                        if same:
                            d3 = A.dot(S, 2) * psi - psi * A.dot(T, 1)
                            d4 = psi * A.dot(T, 2) - A.dot(S, 1) * psi
                            n_checks += 2
                            if d3 != -rr:
                                fails.append(("x2 psi - psi x1", s, d3, -rr))
                            if d4 != -rr:
                                fails.append(("psi x2 - x1 psi", s, d4, -rr))
                else:
                    got = A.word(S, (1, 2, 1)) - A.word(S, (2, 1, 2))
                    exp = expected_cubic(A, s)
                    n_checks += 1
                    if got != exp:
                        fails.append(("cubic", s, got, exp))
                    # dots far from a crossing commute with it
                    for i, j in ((3, 1), (1, 2)):
                        if is_red(s[i - 1]):
                            continue
                        t = act(s, j)
                        lhs = A.dot(S, i) * A.x(S, j)
                        rhs = A.x(S, j) * A.dot(seqstr(t), i)
                        n_checks += 1
                        if lhs != rhs:
                            fails.append(("far dot", s, i, j))
        # far commutation on 4-strand states
        for s in [(1, 2, 1, 2), (1, 1, 2, 2), (2, "W%d" % k, k, 1), ("W%d" % k, k, k, 2), (k, k, "W%d" % k, k)]:
            A = Alg(CT, seqstr(s))
            S = seqstr(s)
            n_checks += 1
            if A.word(S, (1, 3)) != A.word(S, (3, 1)):
                fails.append(("far psi", s))
    print("relation checks:", n_checks, "  failures:", len(fails))
    for f in fails:
        print("   FAIL", f)
    # print the table of cubic values (non-zero ones), for the record
    print("non-zero cubic terms psi1 psi2 psi1 - psi2 psi1 psi2:")
    for k in (1, 2):
        for s in itertools.product([1, 2, 3, "W%d" % k], repeat=3):
            if sum(is_red(a) for a in s) > 1:
                continue
            A = Alg(CT, seqstr(s))
            v = A.word(seqstr(s), (1, 2, 1)) - A.word(seqstr(s), (2, 1, 2))
            if v != 0 * v:
                print("   ", seqstr(s), ":", v)


if __name__ == "__main__":
    main()
