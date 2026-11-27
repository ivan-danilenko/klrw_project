"""
R01_klrw.py  (referee of 01_foundations.md; own code)

    sage -python proofs/checks/R01_klrw.py [basis|hom|assoc|embedded|grading|all] [seed]

Is Definition A.1 what `klrw` implements, beyond the 2-/3-strand patterns of 01 §A.2?

  basis    : for every basis braid b of klrw (states up to 6 strands), the product of the
             generators psi_i along the word of b, computed in klrw letter by letter, is the
             basis element b itself (no lower terms), and b's word is the Lehmer-cocode word of
             01 Def. B.10 (own implementation).
  hom      : rho(a*b) = rho(b) o rho(a) for the referee's own polynomial representation
             (R01_lib.PolyRepR), random dotted basis elements a, b on 4-6 strand states with the
             red strand anywhere, parameters specialised at random in GF(p).
  assoc    : (a*b)*c = a*(b*c) in klrw for random dotted basis elements on 4-6 strands.
  embedded : the relations (A3)-(A6) placed at every position i of a 5-6 strand state, with
             a random dotted braid in front and behind, multiplied letter by letter.
  grading  : every product of homogeneous elements is homogeneous of the sum degree, and the
             klrw degrees of x, psi, r, t, u agree with the table F2 (PROOFS.md 1.2).
"""
import itertools
import random
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from R01_lib import (is_red, seq_str, parse_seq, node_tok, PolyRepR, random_params,  # noqa
                     act_klrw, compose_act, lehmer_word, states_of)
from alg import Alg  # noqa
from sage.all import GF, ZZ  # noqa

P_PRIME = 1000003

CASES = [
    (["A", 2], "1 1 2 W1 1"),
    (["A", 2], "1 2 1 W1 2 1"),
    (["A", 2], "2 1 1 W2 2"),
    (["A", 3], "1 2 3 2 W2"),
    (["A", 3], "2 1 W2 2 3 2"),
    (["D", 4], "2 1 3 W2 4 2"),
    (["A", 2], "1 1 1 W1 2"),
]


def nodes_of(ct):
    return list(range(1, ct[1] + 1))


def rand_elem(A, s, rng, maxdots=2):
    K = A.K
    st = A.state(seq_str(s))
    braids = list(K.KLRWBraid._braids_with_left_state_iter_(st))
    b = rng.choice(braids)
    el = K.monomial(b)
    blacks = [q + 1 for q in range(len(s)) if not is_red(s[q])]
    for _ in range(rng.randint(0, maxdots)):
        el = A.dot(seq_str(s), rng.choice(blacks)) * el
    return el, b


def right_tok_state(b):
    return tuple(node_tok(x) for x in b.right_state())


def test_basis(rng):
    n_ok = n = 0
    n_lehmer = 0
    for ct, seq in CASES:
        A = Alg(ct, seq)
        K = A.K
        s0 = parse_seq(seq)
        for s in states_of(s0):
            st = A.state(seq_str(s))
            braids = list(K.KLRWBraid._braids_with_left_state_iter_(st))
            sample = braids if len(braids) <= 60 else rng.sample(braids, 60)
            for b in sample:
                w = tuple(b.word())
                el = A.word(seq_str(s), w)
                n += 1
                if el == K.monomial(b):
                    n_ok += 1
                else:
                    print("  basis word product != basis element", seq_str(s), w)
                # Lehmer word: compute the permutation of b from the word
                cur = list(range(len(s)))
                for i in w:
                    cur[i - 1], cur[i] = cur[i], cur[i - 1]
                perm = [None] * len(s)
                for pos, src in enumerate(cur):
                    perm[src] = pos
                if tuple(x + 1 for x in lehmer_word(perm)) == w:
                    n_lehmer += 1
                else:
                    print("  klrw word is not the Lehmer word", w, lehmer_word(perm))
    print("basis: %d/%d basis braids equal the letter-by-letter product of their word; "
          "%d/%d are the Lehmer-cocode word of Def. B.10" % (n_ok, n, n_lehmer, n))
    return n_ok == n == n_lehmer


def test_hom(rng, n_pairs=40):
    fails = n = 0
    for ct, seq in CASES:
        A = Alg(ct, seq)
        s0 = parse_seq(seq)
        params = random_params(ct, rng, GF(P_PRIME), nodes_of(ct))
        rep = PolyRepR(ct, len(s0), params, GF(P_PRIME))
        states = states_of(s0)
        for _ in range(n_pairs):
            s = rng.choice(states)
            a, ba = rand_elem(A, s, rng)
            t = right_tok_state(ba)
            b, bb = rand_elem(A, t, rng)
            ab = a * b
            blacks = [q for q in range(len(s)) if not is_red(s[q])]
            for _ in range(2):
                f = rep.R(1)
                for _ in range(rng.randint(0, 3)):
                    f *= rep.y[rng.choice(blacks)] + rep.B(rng.randint(0, 5))
                lhs = act_klrw(A, rep, s, ab, f) if not ab.is_zero() else {}
                rhs = compose_act(A, rep, s, a, b, f)
                n += 1
                if lhs != rhs:
                    fails += 1
                    print("  FAIL hom", seq_str(s), a, b, f)
    print("hom: rho(a*b) = rho(b) o rho(a): %d instances on %d cases, %d failures" % (n, len(CASES), fails))
    return fails == 0


def test_assoc(rng, n_triples=25):
    fails = n = 0
    for ct, seq in CASES:
        A = Alg(ct, seq)
        s0 = parse_seq(seq)
        states = states_of(s0)
        for _ in range(n_triples):
            s = rng.choice(states)
            a, ba = rand_elem(A, s, rng)
            b, bb = rand_elem(A, right_tok_state(ba), rng)
            c, bc = rand_elem(A, right_tok_state(bb), rng)
            n += 1
            if (a * b) * c != a * (b * c):
                fails += 1
                print("  FAIL assoc", seq_str(s))
    print("assoc: %d triples, %d failures" % (n, fails))
    return fails == 0


def C_of(A, a, b, c):
    if is_red(a) or is_red(c) or a != c:
        return 0
    if is_red(b):
        return A.r(a) * A.u(a) if b[1] == a else 0
    from R01_lib import dynkin_edges
    return None  # filled by caller (needs adjacency)


def test_embedded(rng, n_per_state=3):
    from R01_lib import dynkin_edges
    fails = n = 0
    for ct, seq in CASES:
        E = dynkin_edges(ct)
        A = Alg(ct, seq)
        K = A.K
        s0 = parse_seq(seq)
        L = len(s0)
        states = states_of(s0)
        for s in rng.sample(states, min(len(states), 12)):
            for _ in range(n_per_state):
                # random prefix ending at s, random suffix
                # prefix: element with right state s -> build from a random left state
                pre_left = rng.choice(states)
                cand = [b for b in K.KLRWBraid._braids_with_left_state_iter_(A.state(seq_str(pre_left)))
                        if right_tok_state(b) == s]
                if not cand:
                    continue
                pre = K.monomial(rng.choice(cand))
                blacks = [q + 1 for q in range(L) if not is_red(pre_left[q])]
                for _ in range(rng.randint(0, 2)):
                    pre = A.dot(seq_str(pre_left), rng.choice(blacks)) * pre

                def run(word, dots_before=(), dots_after=()):
                    """pre * (dots at s) * psi_word * (dots at target), letter by letter."""
                    el = pre
                    st = s
                    for p in dots_before:
                        el = el * A.dot(seq_str(st), p)
                    for i in word:
                        el = el * A.x(seq_str(st), i)
                        st = list(st)
                        st[i - 1], st[i] = st[i], st[i - 1]
                        st = tuple(st)
                    for p in dots_after:
                        el = el * A.dot(seq_str(st), p)
                    return el, st

                for i in range(1, L):
                    a, b = s[i - 1], s[i]
                    # (A5)
                    el, _ = run((i, i))
                    if is_red(a) and is_red(b):
                        continue
                    if not is_red(a) and not is_red(b):
                        if a == b:
                            Q = 0
                        elif frozenset((a, b)) in E:
                            Q = A.t(a, b) * A.dot(seq_str(s), i) + A.t(b, a) * A.dot(seq_str(s), i + 1)
                        else:
                            Q = A.e(seq_str(s))
                    elif is_red(b):
                        Q = A.u(a) * A.dot(seq_str(s), i) if a == b[1] else A.e(seq_str(s))
                    else:
                        Q = A.u(b) * A.dot(seq_str(s), i + 1) if b == a[1] else A.e(seq_str(s))
                    n += 1
                    if el != pre * Q:
                        fails += 1
                        print("  FAIL A5 embedded", seq_str(s), i)
                    # (A4)
                    same = (not is_red(a)) and a == b
                    rr = A.r(a) if same else 0
                    if not is_red(a):
                        e1, _ = run((i,), dots_before=(i,))
                        e2, _ = run((i,), dots_after=(i + 1,))
                        n += 1
                        if e1 - e2 != rr * pre:
                            fails += 1
                            print("  FAIL A4a embedded", seq_str(s), i)
                    if not is_red(b):
                        e1, _ = run((i,), dots_after=(i,))
                        e2, _ = run((i,), dots_before=(i + 1,))
                        n += 1
                        if e1 - e2 != rr * pre:
                            fails += 1
                            print("  FAIL A4b embedded", seq_str(s), i)
                    # (A3) far dots
                    for p in range(1, L + 1):
                        if p in (i, i + 1) or is_red(s[p - 1]):
                            continue
                        e1, _ = run((i,), dots_before=(p,))
                        e2, _ = run((i,), dots_after=(p,))
                        n += 1
                        if e1 != e2:
                            fails += 1
                            print("  FAIL A3 embedded", seq_str(s), i, p)
                    # (A2)
                    for j in range(i + 2, L):
                        e1, _ = run((i, j))
                        e2, _ = run((j, i))
                        n += 1
                        if e1 != e2:
                            fails += 1
                            print("  FAIL A2 embedded", seq_str(s), i, j)
                    # (A6)
                    if i <= L - 2:
                        c = s[i + 1]
                        e1, _ = run((i, i + 1, i))
                        e2, _ = run((i + 1, i, i + 1))
                        C = 0
                        if (not is_red(a)) and (not is_red(c)) and a == c:
                            if is_red(b):
                                C = A.r(a) * A.u(a) if b[1] == a else 0
                            elif frozenset((a, b)) in E:
                                C = A.r(a) * A.t(a, b)
                        n += 1
                        if e1 - e2 != C * pre:
                            fails += 1
                            print("  FAIL A6 embedded", seq_str(s), i, e1 - e2)
    print("embedded: %d relation instances (behind a random dotted braid, letter by letter), %d failures"
          % (n, fails))
    return fails == 0


def test_grading(rng, n_pairs=30):
    fails = n = 0
    for ct, seq in CASES[:4]:
        A = Alg(ct, seq)
        s0 = parse_seq(seq)
        states = states_of(s0)
        for _ in range(n_pairs):
            s = rng.choice(states)
            a, ba = rand_elem(A, s, rng)
            b, bb = rand_elem(A, right_tok_state(ba), rng)
            ab = a * b
            n += 1
            if ab.is_zero():
                continue
            try:
                d = ab.degree(check_if_homogeneous=True)
            except ValueError:
                fails += 1
                print("  FAIL: product not homogeneous", seq_str(s))
                continue
            if d != a.degree(check_if_homogeneous=True) + b.degree(check_if_homogeneous=True):
                fails += 1
                print("  FAIL: degree not additive", seq_str(s))
    # generator degrees against F2
    A = Alg(["A", 2], "1 1 2 W1")
    gg = A.K.grading_group
    print("  klrw degrees (A_2, red 1):")
    for name, el in [("psi(1,1)", A.x("1 1 2 W1", 1)), ("psi(1,2)", A.x("1 2 1 W1", 2)),
                     ("psi(1,W1)", A.x("1 1 W1 2", 2)), ("psi(2,W1)", A.x("1 1 2 W1", 3)),
                     ("x on 1", A.dot("1 1 2 W1", 1)), ("x on 2", A.dot("1 1 2 W1", 3)),
                     ("r_1", A.r(1) * A.e("1 1 2 W1")), ("t_12", A.t(1, 2) * A.e("1 1 2 W1")),
                     ("t_21", A.t(2, 1) * A.e("1 1 2 W1")), ("u_1", A.u(1) * A.e("1 1 2 W1"))]:
        print("    %-10s %s" % (name, el.degree()))
    print("grading: %d products, %d failures" % (n, fails))
    return fails == 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng = random.Random(seed)
    ok = True
    if mode in ("basis", "all"):
        ok &= test_basis(rng)
    if mode in ("hom", "all"):
        ok &= test_hom(rng)
    if mode in ("assoc", "all"):
        ok &= test_assoc(rng)
    if mode in ("embedded", "all"):
        ok &= test_embedded(rng)
    if mode in ("grading", "all"):
        ok &= test_grading(rng)
    print("ALL OK" if ok else "SOME FAILURES")
