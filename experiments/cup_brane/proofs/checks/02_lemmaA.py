"""
Lemma 2.3 of proofs/02_omega1_ibrane.md (product of reduced diagrams when all black colours are
distinct), checked in the KLRW algebra of klrw/.

    sage -python proofs/checks/02_lemmaA.py N SAMPLES [SEED]

Algebra: type A_N, red strands W_1 and W_N (for N = 1: two red strands of colour 1), one black
strand of each colour 1..N; all three scalings on, polynomial parameters (not inverted), base ZZ.
A *state* is an ordering of the labels 1..N, Wa (= W_1), Wb (= W_N) with Wa left of Wb.

For random triples of states (s, s', s'') (all triples if there are few) we check

    beta(s, s') * beta(s', s'')  ==  prod_{pairs crossing twice} Q_pair * beta(s, s'')

where beta(s, s') is the KLRW basis braid between the two states (klrw's choice of reduced word),
Q_{c,c+1} = t_{c,c+1} x_c + t_{c+1,c} x_{c+1}, Q_{Wa,1} = u_1 x_1, Q_{N,Wb} = u_N x_N, and every
other pair has Q = 1.  Also: beta(s, s') does not depend on the reduced word (checked by
comparing klrw's word with a second, independently constructed reduced word: bubble sort from
the right instead of from the left).
"""
import itertools
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from cupbrane import *  # noqa
from klrw.stable_envelopes import braid_from_marked_state


def setup(n):
    ct = CartanType(["A", n])
    DD = FramedDynkinDiagram_with_dimensions(ct)
    Wa = NodeInFramedQuiver(1, framing=True)
    Wb = NodeInFramedQuiver(n, framing=True)
    DD[Wa] += 1
    DD[Wb] += 1
    for c in range(1, n + 1):
        DD[NodeInFramedQuiver(c)] += 1
    K = KLRWAlgebra(ZZ, DD, dot_scaling=True, edge_scaling=True, vertex_scaling=True,
                    invertible_parameters=False, warnings=True)
    return K, Wa, Wb


def main(n, samples, seed=0):
    K, Wa, Wb = setup(n)
    Dring = K.base()
    Braid = K.KLRWBraid
    State = Braid.KLRWstate_set
    names = Dring.variable_names()

    def node(x):
        return Wa if x == "Wa" else Wb if x == "Wb" else NodeInFramedQuiver(x)

    def state(s):
        return State._element_constructor_(tuple(node(x) for x in s))

    def beta(s, s2, word=None):
        if word is None:
            word = braid_from_marked_state(tuple(s), tuple(s2))
            return K.monomial(Braid._element_constructor_(state(s2), tuple(word)))
        # an arbitrary word: product of single crossings, first letter acting first on s
        return K.element_from_braid(tuple(word), left_state=state(s))

    def other_word(s, s2):
        """A reduced word (letters = positions, first letter first, read from s) by bubble sort,
        moving the strands into place from the right; in general different from klrw's normal form."""
        cur = list(s)
        word = []
        target = list(s2)
        for pos in range(len(cur) - 1, -1, -1):
            lab = target[pos]
            i = cur.index(lab)
            while i < pos:
                cur[i], cur[i + 1] = cur[i + 1], cur[i]
                word.append(i + 1)
                i += 1
        assert cur == target
        return word

    def dot(c):
        return Dring("x_%d_1" % c)

    def par(nm):
        return Dring(nm)

    def Q(a, b):
        """Bigon polynomial of the unordered pair {a, b}."""
        pair = {a, b}
        if pair == {"Wa", 1}:
            return par("u_1") * dot(1)
        if pair == {n, "Wb"}:
            return par("u_%d" % n) * dot(n)
        if all(isinstance(x, int) for x in pair) and abs(a - b) == 1:
            lo, hi = min(a, b), max(a, b)
            return par("t_%d_%d" % (lo, hi)) * dot(lo) + par("t_%d_%d" % (hi, lo)) * dot(hi)
        return Dring.one()

    labels = list(range(1, n + 1)) + ["Wa", "Wb"]
    states = [p for p in itertools.permutations(labels) if p.index("Wa") < p.index("Wb")]

    def left(s, a, b):
        return s.index(a) < s.index(b)

    rng = random.Random(seed)
    if len(states) ** 3 <= samples:
        triples = list(itertools.product(states, repeat=3))
    else:
        triples = [tuple(rng.choice(states) for _ in range(3)) for _ in range(samples)]
    bad = 0
    nontrivial = 0
    words_bad = 0
    words_differ = 0
    for s, s1, s2 in triples:
        lhs = beta(s, s1) * beta(s1, s2)
        f = Dring.one()
        for a, b in itertools.combinations(labels, 2):
            if left(s, a, b) != left(s1, a, b) and left(s1, a, b) != left(s2, a, b):
                f *= Q(a, b)
        rhs = f * beta(s, s2)
        if f != Dring.one():
            nontrivial += 1
        if lhs != rhs:
            bad += 1
            if bad <= 3:
                print("MISMATCH", s, s1, s2, lhs, rhs)
        w2 = other_word(s, s1)
        if tuple(w2) != tuple(braid_from_marked_state(tuple(s), tuple(s1))):
            words_differ += 1
        if beta(s, s1) != beta(s, s1, w2):
            words_bad += 1
    print("n=%d: %d states, %d triples checked (%d with a non-trivial bigon factor): mismatches %d;"
          " second reduced word differs from klrw's in %d cases, gives a different element in %d"
          % (n, len(states), len(triples), nontrivial, bad, words_differ, words_bad), flush=True)

    # --- local relations on EVERY state (Lemma 2.1): bigons and braid relations --------------
    bad_bigon = bad_braid = n_braid = 0
    for s in states:
        for i in range(1, len(s)):
            a, b = s[i - 1], s[i]
            if {a, b} == {"Wa", "Wb"}:
                continue                                     # red strands never cross
            if K.element_from_braid((i, i), left_state=state(s)) != Q(a, b) * K.idempotent(state(s)):
                bad_bigon += 1
        for i in range(1, len(s) - 1):
            if "Wa" in s[i - 1:i + 2] and "Wb" in s[i - 1:i + 2]:
                continue                                     # would need a red-red crossing
            n_braid += 1
            if K.element_from_braid((i, i + 1, i), left_state=state(s)) != \
                    K.element_from_braid((i + 1, i, i + 1), left_state=state(s)):
                bad_braid += 1
    print("      local relations on all %d states: bigon mismatches %d; braid relation failures %d (of %d triples)"
          % (len(states), bad_bigon, bad_braid, n_braid), flush=True)

    # --- degree of beta(s, s') (Proposition 6.1): sum over flipped edges of E_x = q + e_x ------
    G = K.grading_group

    def E_vec(x):
        """E_x = deg(Q_x) / 2, Q_x the bigon polynomial of edge x."""
        lab_a = "Wa" if x == 0 else x
        lab_b = "Wb" if x == n else x + 1
        qx = Q(lab_a, lab_b)
        mono = qx.monomials()[0]
        v = Dring.element_degree(mono, G)._vector_()
        return v / 2

    def order_bit(s, x):
        lab_a = "Wa" if x == 0 else x
        lab_b = "Wb" if x == n else x + 1
        return left(s, lab_a, lab_b)

    bad_deg = 0
    pairs = [(rng.choice(states), rng.choice(states)) for _ in range(min(samples, 400))]
    for s, s1 in pairs:
        expect = sum((E_vec(x) for x in range(n + 1) if order_bit(s, x) != order_bit(s1, x)),
                     0 * E_vec(0))
        got = beta(s, s1).degree(check_if_homogeneous=True)._vector_()
        if got != expect:
            bad_deg += 1
    print("      degree of beta = sum of E_x over flipped edges: %d pairs, mismatches %d" % (len(pairs), bad_deg),
          flush=True)
    return bad == 0 and words_bad == 0 and bad_bigon == 0 and bad_braid == 0 and bad_deg == 0


if __name__ == "__main__":
    n = int(sys.argv[1])
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(n, samples, seed)
