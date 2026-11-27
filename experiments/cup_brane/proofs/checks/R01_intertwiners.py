"""
R01_intertwiners.py  (referee of 01_foundations.md; own code)

    sage -python proofs/checks/R01_intertwiners.py [d1|d3|d5|d6|all]

  d1 : Lemma D.1 on all 2-strand states.
  d3 : Theorem D.3 (exact braid relation for phi) on every 3-strand state over {1,2,3, W_k}
       (A_3, k = 1, 2), and embedded at every position of 5-strand states behind a random
       dotted braid.
  d5 : Proposition D.5, both identities, all 3-strand states.
  d6 : Theorem D.6 (naturality), both directions, for random elements a (random dotted,
       generally non-reduced words) of blocks of 3-4 strands (with and without the red
       strand) and S of every colour; negative control with psi in place of phi.
"""
import itertools
import random
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from R01_lib import is_red, seq_str, parse_seq  # noqa
from alg import Alg  # noqa


def swap(s, i):
    s = list(s)
    s[i - 1], s[i] = s[i], s[i - 1]
    return tuple(s)


class Phi:
    def __init__(self, A, use_phi=True):
        self.A = A
        self.use_phi = use_phi

    def phi(self, s, i):
        """phi_{i,s} (1-based i), element of e(s) A e(s_i s)."""
        A = self.A
        a, b = s[i - 1], s[i]
        psi = A.x(seq_str(s), i)
        if self.use_phi and (not is_red(a)) and a == b:
            return (A.dot(seq_str(s), i) - A.dot(seq_str(s), i + 1)) * psi - A.r(a) * A.e(seq_str(s))
        return psi

    def word(self, s, w):
        el = self.A.e(seq_str(s))
        for i in w:
            el = el * self.phi(s, i)
            s = swap(s, i)
        return el, s


def d1():
    ok = True
    n = 0
    for k in (1, 2):
        for s in itertools.product([1, 2, 3, ("W", k)], repeat=2):
            if all(is_red(x) for x in s):
                continue
            A = Alg(["A", 3], seq_str(s))
            P = Phi(A)
            ph = P.phi(s, 1)
            t = swap(s, 1)
            a, b = s
            if (not is_red(a)) and a == b:
                x1, x2 = A.dot(seq_str(s), 1), A.dot(seq_str(s), 2)
                psi = A.x(seq_str(s), 1)
                r = A.r(a)
                conds = [ph == -psi * (x1 - x2) + r * A.e(seq_str(s)), x1 * ph == ph * x2, x2 * ph == ph * x1,
                         ph * ph == r * r * A.e(seq_str(s))]
            else:
                conds = []
                if not is_red(a):
                    conds.append(A.dot(seq_str(s), 1) * ph == ph * A.dot(seq_str(t), 2))
                if not is_red(b):
                    conds.append(A.dot(seq_str(s), 2) * ph == ph * A.dot(seq_str(t), 1))
            n += len(conds)
            if not all(conds):
                ok = False
                print("  FAIL D.1", s)
    print("d1: %d identities, %s" % (n, "all hold" if ok else "FAILURES"))
    return ok


def three_states(k):
    for s in itertools.product([1, 2, 3, ("W", k)], repeat=3):
        if sum(is_red(x) for x in s) <= 1:
            yield s


def d3(rng):
    ok = True
    n = 0
    for k in (1, 2):
        for s in three_states(k):
            A = Alg(["A", 3], seq_str(s))
            P = Phi(A)
            e1, t1 = P.word(s, (1, 2, 1))
            e2, t2 = P.word(s, (2, 1, 2))
            n += 1
            if e1 != e2 or t1 != t2:
                ok = False
                print("  FAIL D.3", s)
    # embedded in 5 strands behind a random dotted braid
    m = 0
    for seq in ["1 1 1 W1 2", "1 2 1 1 W1", "2 1 W1 1 1", "1 W1 1 2 1"]:
        s0 = parse_seq(seq)
        A = Alg(["A", 2], seq)
        P = Phi(A)
        K = A.K
        states = sorted(set(itertools.permutations(s0)), key=str)
        for s in rng.sample(states, min(8, len(states))):
            left = rng.choice(states)
            cand = [b for b in K.KLRWBraid._braids_with_left_state_iter_(A.state(seq_str(left)))
                    if tuple((("W", x.node) if x.is_framing() else x.node) for x in b.right_state()) == s]
            if not cand:
                continue
            pre = K.monomial(rng.choice(cand))
            bl = [q + 1 for q in range(len(left)) if not is_red(left[q])]
            pre = A.dot(seq_str(left), rng.choice(bl)) * pre
            for i in range(1, len(s) - 1):
                e1, _ = P.word(s, (i, i + 1, i))
                e2, _ = P.word(s, (i + 1, i, i + 1))
                m += 1
                if pre * e1 != pre * e2:
                    ok = False
                    print("  FAIL D.3 embedded", s, i)
    print("d3: %d 3-strand states, %d embedded instances: %s" % (n, m, "all exact" if ok else "FAILURES"))
    return ok


def d5():
    ok = True
    n = 0
    for k in (1, 2):
        for s in three_states(k):
            A = Alg(["A", 3], seq_str(s))
            P = Phi(A)
            # first identity on (A,B,S): psi_1 phi_2 phi_1 = phi_2 phi_1 psi_2
            l = A.x(seq_str(s), 1) * P.word(swap(s, 1), (2, 1))[0]
            r_el, t = P.word(s, (2, 1))
            r_el = r_el * A.x(seq_str(t), 2)
            # second identity on (S,A,B): phi_1 phi_2 psi_1 = psi_2 phi_1 phi_2
            e3, t3 = P.word(s, (1, 2))
            l2 = e3 * A.x(seq_str(t3), 1)
            r2 = A.x(seq_str(s), 2) * P.word(swap(s, 2), (1, 2))[0]
            n += 2
            if l != r_el:
                ok = False
                print("  FAIL D.5 first", s)
            if l2 != r2:
                ok = False
                print("  FAIL D.5 second", s)
    print("d5: %d identities on all 3-strand states: %s" % (n, "all hold" if ok else "FAILURES"))
    return ok


def random_ops(block, rng, L):
    """random generalised word on the block: list of ('x', p) / ('psi', i), 1-based."""
    ops = []
    s = block
    for _ in range(L):
        if rng.random() < 0.35:
            bl = [q + 1 for q in range(len(s)) if not is_red(s[q])]
            ops.append(("x", rng.choice(bl)))
        else:
            i = rng.randint(1, len(s) - 1)
            ops.append(("psi", i))
            s = swap(s, i)
    return ops, s


def build(A, s, ops, shift):
    el = A.e(seq_str(s))
    for kind, p in ops:
        if kind == "x":
            el = el * A.dot(seq_str(s), p + shift)
        else:
            el = el * A.x(seq_str(s), p + shift)
            s = swap(s, p + shift)
    return el, s


def d6(rng, use_phi=True, n_per=12):
    fails = n = 0
    blocks = [(["A", 3], "1 2 1"), (["A", 3], "1 1 2"), (["A", 3], "1 W1 1"), (["A", 3], "2 1 W1"),
              (["A", 3], "1 1 1"), (["A", 3], "W2 2 1"), (["A", 3], "1 2 W1 1"), (["A", 3], "2 1 1 W2")]
    for ct, bseq in blocks:
        b0 = parse_seq(bseq)
        has_red = any(is_red(x) for x in b0)
        colours = [1, 2, 3] + ([] if has_red else [("W", 1), ("W", 2)])
        for c in colours:
            full = b0 + (c,)
            A = Alg(ct, seq_str(full))
            P = Phi(A, use_phi)
            states = sorted(set(itertools.permutations(b0)), key=str)
            nb = len(b0)
            for _ in range(n_per):
                t = rng.choice(states)
                ops, t2 = random_ops(t, rng, rng.randint(1, 5))
                # (a x 1) Phi(t') = Phi(t) (1 x a)
                a1, _ = build(A, t + (c,), ops, 0)
                Phi_t2, _ = P.word(t2 + (c,), tuple(range(nb, 0, -1)))
                Phi_t, _ = P.word(t + (c,), tuple(range(nb, 0, -1)))
                a2, _ = build(A, (c,) + t, ops, 1)
                n += 1
                if a1 * Phi_t2 != Phi_t * a2:
                    fails += 1
                # (1 x a) Phi'(t') = Phi'(t) (a x 1)
                Pp_t2, _ = P.word((c,) + t2, tuple(range(1, nb + 1)))
                Pp_t, _ = P.word((c,) + t, tuple(range(1, nb + 1)))
                n += 1
                if a2 * Pp_t2 != Pp_t * a1:
                    fails += 1
    print("d6 (%s): %d naturality identities, %d failures" % ("phi" if use_phi else "psi, negative control", n, fails))
    return fails == 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    rng = random.Random(17)
    ok = True
    if mode in ("d1", "all"):
        ok &= d1()
    if mode in ("d3", "all"):
        ok &= d3(rng)
    if mode in ("d5", "all"):
        ok &= d5()
    if mode in ("d6", "all"):
        ok &= d6(rng)
        d6(rng, use_phi=False, n_per=4)
    print("ALL OK" if ok else "FAILURES")
