"""
01_intertwiners.py   (sage -python proofs/checks/01_intertwiners.py)

Checks of section D of 01_foundations.md, exactly, in the klrw implementation:

  phi_i e(s) = (x_i - x_{i+1}) psi_i - r_c        if s_i = s_{i+1} = c black
             = psi_i                               otherwise (different colours, or black-red)

 (D1) phi = -psi (x_1 - x_2) + r_c  (equal colours)
 (D2) dots follow their strand through phi:  x_1 phi = phi x_2,  x_2 phi = phi x_1
 (D3) phi^2 = r_c^2 (equal colours), Q_{ab}(x_1, x_2) (otherwise)
 (D4) phi_1 phi_2 phi_1 = phi_2 phi_1 phi_2 on EVERY 3-strand state (colours of A_3 and one red)
 (D5) mixed relation:  psi_1 phi_2 phi_1 = phi_2 phi_1 psi_2  on (A, B, s)   (s moves right->left)
                        phi_1 phi_2 psi_1 = psi_2 phi_1 phi_2  on (s, A, B)   (s moves left->right)
 (D6) naturality: for a strand s at the far right crossing a block by intertwiners,
      Phi = phi_n phi_{n-1} ... phi_1 (first phi_n), and random elements a of the block
      algebra:  (a (x) 1) * Phi = Phi * (1 (x) a).   Also the mirror statement.
"""
import itertools
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alg import Alg   # noqa: E402

ADJ = {frozenset((1, 2)), frozenset((2, 3))}


def is_red(a):
    return isinstance(a, str)


def seqstr(s):
    return " ".join(a if is_red(a) else str(a) for a in s)


def act(s, i):
    s = list(s)
    s[i - 1], s[i] = s[i], s[i - 1]
    return tuple(s)


def phi(A, s, i):
    S = seqstr(s)
    a, b = s[i - 1], s[i]
    if not is_red(a) and a == b:
        return (A.dot(S, i) - A.dot(S, i + 1)) * A.x(S, i) - A.r(a) * A.e(S)
    return A.x(S, i)


def phi_word(A, s, w):
    el = A.e(seqstr(s))
    for i in w:
        el = el * phi(A, s, i)
        s = act(s, i)
    return el, s


def psi_word(A, s, w):
    return A.word(seqstr(s), tuple(w)), tuple(_apply(s, w))


def _apply(s, w):
    for i in w:
        s = act(s, i)
    return s


def Qexp(A, s):
    a, b = s
    S = seqstr(s)
    e = A.e(S)
    if not is_red(a) and not is_red(b):
        if a == b:
            return A.r(a) ** 2 * e
        if frozenset((a, b)) in ADJ:
            return A.t(a, b) * A.dot(S, 1) + A.t(b, a) * A.dot(S, 2)
        return e
    if is_red(b):
        return A.u(a) * A.dot(S, 1) if a == int(b[1:]) else e
    return A.u(b) * A.dot(S, 2) if b == int(a[1:]) else e


def main():
    fails = []
    n = 0
    for k in (1, 2):
        alph = [1, 2, 3, "W%d" % k]
        for s in itertools.product(alph, repeat=2):
            if sum(is_red(a) for a in s) > 1:
                continue
            A = Alg(["A", 3], seqstr(s))
            S, T = seqstr(s), seqstr(act(s, 1))
            p = phi(A, s, 1)
            if not is_red(s[0]) and s[0] == s[1]:
                n += 1
                alt = -A.x(S, 1) * (A.dot(T, 1) - A.dot(T, 2)) + A.r(s[0]) * A.e(S)
                if p != alt:
                    fails.append(("D1", s))
            if not is_red(s[0]):
                n += 1
                if A.dot(S, 1) * p != p * A.dot(T, 2):
                    fails.append(("D2 x1", s))
            if not is_red(s[1]):
                n += 1
                if A.dot(S, 2) * p != p * A.dot(T, 1):
                    fails.append(("D2 x2", s))
            n += 1
            if p * phi(A, act(s, 1), 1) != Qexp(A, s):
                fails.append(("D3", s, p * phi(A, act(s, 1), 1)))
        for s in itertools.product(alph, repeat=3):
            if sum(is_red(a) for a in s) > 1:
                continue
            A = Alg(["A", 3], seqstr(s))
            l1, t1 = phi_word(A, s, (1, 2, 1))
            l2, t2 = phi_word(A, s, (2, 1, 2))
            n += 1
            if l1 != l2:
                fails.append(("D4", s, l1 - l2))
            # (D5) s = third strand moves right -> left past (A, B)
            # psi_1 phi_2 phi_1  vs  phi_2 phi_1 psi_2
            lhs = A.x(seqstr(s), 1) * phi_word(A, act(s, 1), (2, 1))[0]
            m, st = phi_word(A, s, (2, 1))
            rhs = m * A.x(seqstr(st), 2)
            n += 1
            if lhs != rhs:
                fails.append(("D5 right->left", s, lhs - rhs))
            # s = first strand moves left -> right past (A, B):  phi_1 phi_2 psi_1 = psi_2 phi_1 phi_2
            m1, st1 = phi_word(A, s, (1, 2))
            lhs = m1 * A.x(seqstr(st1), 1)
            rhs = A.x(seqstr(s), 2) * phi_word(A, act(s, 2), (1, 2))[0]
            n += 1
            if lhs != rhs:
                fails.append(("D5 left->right", s, lhs - rhs))
    print("intertwiner identities (D1)-(D5): %d checks, %d failures" % (n, len(fails)))
    for f in fails:
        print("   FAIL", f)
    return fails


def naturality(n_tests=25, seed=3):
    """(D6) on blocks of 3 strands (one may be red) and a strand s of each colour."""
    rng = random.Random(seed)
    fails = 0
    n = 0
    blocks = [(1, 2, 1), (1, 1, 2), (1, "W1", 1), (2, 1, "W1"), (1, 1, 1), ("W2", 2, 1)]
    for blk in blocks:
        for c in (1, 2, 3):
            full = blk + (c,)
            A = Alg(["A", 3], seqstr(full))
            K = A.K
            nb = len(blk)
            states = sorted(set(itertools.permutations(blk)), key=str)
            for _ in range(n_tests):
                t = rng.choice(states)
                # random element of the block algebra starting at t: random word (not nec. reduced) with dots
                w = [rng.randint(1, nb - 1) for _ in range(rng.randint(0, 4))]
                el_b = A.e(seqstr(t + (c,)))
                cur = t
                for i in w:
                    if rng.random() < 0.4:
                        blacks = [q + 1 for q in range(nb) if not is_red(cur[q])]
                        el_b = el_b * A.dot(seqstr(cur + (c,)), rng.choice(blacks))
                    el_b = el_b * A.x(seqstr(cur + (c,)), i)
                    cur = act(cur, i)
                t2 = cur
                # a (x) 1 : el_b (block at positions 1..nb, s at nb+1);   1 (x) a : shift positions by one
                el_s = A.e(seqstr((c,) + t))
                cur = t
                # rebuild the same element on the shifted state (dots and crossings shifted by 1)
                rng2 = None
                # Phi from (t2, c) to (c, t2): phi_nb ... phi_1
                Phi_t2, st_end = phi_word(A, t2 + (c,), tuple(range(nb, 0, -1)))
                Phi_t, st0 = phi_word(A, t + (c,), tuple(range(nb, 0, -1)))
                assert st_end == (c,) + t2 and st0 == (c,) + t
                lhs = el_b * Phi_t2
                rhs = Phi_t * shift(A, el_b, c)
                n += 1
                if lhs != rhs:
                    fails += 1
                    print("FAIL naturality", blk, c, t, w)
    print("naturality (a (x) 1) Phi = Phi (1 (x) a): %d random elements, %d failures" % (n, fails))
    return fails


def shift(A, el, c):
    """The element 1_c (x) el: el (living on states (t, c)) moved to the states (c, t)."""
    K = A.K
    out = None
    for braid, coeff in el:
        left = tuple(_tok(x) for x in braid.left_state())
        assert left[-1] == c
        t = left[:-1]
        word = tuple(i + 1 for i in braid.word())
        # coefficient: dots of the left state (t, c) -> same strands in (c, t);
        # the colour-c index of a strand shifts by one if c equals its colour (the new c strand is first)
        D = A.D
        names = D.variable_names()
        new = A.e(seqstr((c,) + t))
        poly = None
        term = A.word(seqstr((c,) + t), word)
        for exp, cf in coeff.iterator_exp_coeff():
            mono = A.e(seqstr((c,) + t)) * cf
            for pos, power in exp.sparse_iter():
                name = names[pos]
                if name.startswith("x_"):
                    _, col, idx = name.split("_")
                    col, idx = int(col), int(idx)
                    p = [q for q in range(len(t)) if t[q] == col][idx - 1]   # position in t (0-based)
                    for _ in range(power):
                        mono = mono * A.dot(seqstr((c,) + t), p + 2)
                else:
                    mono = mono * (D(name) ** power)
            poly = mono if poly is None else poly + mono
        piece = poly * term
        out = piece if out is None else out + piece
    return out


def _tok(node):
    return "W%d" % node.node if node.is_framing() else node.node


if __name__ == "__main__":
    main()
    naturality()
