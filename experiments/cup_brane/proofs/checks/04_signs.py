"""
Pure-Python checks of the sign bookkeeping in proofs/04_first_order.md (Lemma 5.1 and the
sign identities in the proof of Theorem C).  No algebra involved.

    python3 proofs/checks/04_signs.py TYPE RANK MAXLEN

For all words of length <= MAXLEN and all vertices I:
  (L5.1)  the sign of every one-pair term of formula.py equals (-1)^(n_I(j) + N_I(a,b));
  (C1)    bad triple (y, i, j):  sign of the composite  = - eps(I; i, j);
  (C2)    generic (x, j, p):     s_A = - s_B;
  (C3)    if also a < x < b and c(x) ~ c(p):  s_A' = - s_A   (roles of x and j exchanged).
"""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from formula import Stack  # noqa: E402


def n_above(I, x):
    return sum(1 for z in I if z > x)


def N_between(I, a, b):
    return sum(1 for z in I if a < z < b)


def sigma1(I, j, p):
    return (-1) ** (n_above(I, j) + N_between(I, p[0], p[1]))


def one_pair_terms(st, I):
    out = {}
    for J, sigma, mono, sign in st.terms_from(frozenset(I)):
        if sum(pw for nm, pw in mono if nm.startswith("r_")) != 1:
            continue
        moved = [s for s in range(1, st.m + 1) if sigma[s - 1] != s]
        a, b = sorted(moved)
        (j,) = [s for s in frozenset(I) - J if s != b]
        out[j, (a, b)] = sign
    return out


def check(ct, word):
    st = Stack(ct, word)
    m = st.m
    bad = 0
    cnt = [0, 0, 0, 0]
    allI = [frozenset(c) for h in range(m + 1) for c in itertools.combinations(range(1, m + 1), h)]
    terms = {I: one_pair_terms(st, I) for I in allI}
    for I in allI:
        for (j, p), sign in terms[I].items():
            cnt[0] += 1
            if sign != sigma1(I, j, p):
                print("L5.1 FAIL", word, sorted(I), j, p); bad += 1
        # bad triples
        for i, j in itertools.combinations(sorted(I), 2):
            if not st.adjacent(i, j):
                continue
            eps = (-1) ** (n_above(I, i) + n_above(I, j))
            for y in range(1, i):
                if st.col[y - 1] != st.col[j - 1]:
                    continue
                cnt[1] += 1
                if y in I:      # (A1): m_y(I) then T(I - y; i, (y, j))
                    s = (-1) ** n_above(I, y) * terms[I - {y}][i, (y, j)]
                else:           # (B1): T(I; i, (y, j)) then m_y(J)
                    J = (I - {i, j}) | {y}
                    s = terms[I][i, (y, j)] * (-1) ** n_above(J, y)
                if s != -eps:
                    print("C1 FAIL", word, sorted(I), y, i, j); bad += 1
        # generic composites
        for (j, p), sign in terms[I].items():
            a, b = p
            J = (I - {j, b}) | {a}
            for x in I - {j, b}:
                cnt[2] += 1
                sA = (-1) ** n_above(I, x) * terms[I - {x}][j, p]
                sB = sign * (-1) ** n_above(J, x)
                if sA != -sB:
                    print("C2 FAIL", word, sorted(I), x, j, p); bad += 1
                if a < x < b and st.adjacent(x, a):
                    cnt[3] += 1
                    sA2 = (-1) ** n_above(I, j) * terms[I - {j}][x, p]
                    if sA2 != -sA:
                        print("C3 FAIL", word, sorted(I), x, j, p); bad += 1
    return bad, cnt


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    maxlen = int(sys.argv[3])
    tot = [0, 0, 0, 0]; nbad = 0; nw = 0
    for L in range(1, maxlen + 1):
        for w in itertools.product(range(1, ct[1] + 1), repeat=L):
            b, c = check(ct, w)
            nbad += b; nw += 1
            tot = [x + y for x, y in zip(tot, c)]
    print("type", ct, "words", nw, "failures", nbad, "| one-pair terms %d, bad triples %d, generic (x,j,p) %d, swapped %d" % tuple(tot))
