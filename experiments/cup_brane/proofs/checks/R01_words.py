"""
R01_words.py  (referee of 01_foundations.md; own code)

    sage -python proofs/checks/R01_words.py [b7|b5|b8|c3|all]

  b7 : §B.7: A_2, red 1, word 11221121, I={3,5,6,7,8}, mover 3, F={(1,8),(2,5),(4,7)}:
       random reduced words of sigma give different elements; the variant with (2,6) does not.
  b5 : §B.5 example (cycle-free, bad but shielded): A_3, word 12321223, I={4,...,8}, mover 7,
       F={(1,5),(2,6),(3,8)}, red 1,2,3: all random reduced words give one element.
  b8 : Lemma B.8: the six reduced words of 4231 on (c,c',c,c), (c,c,c',c), the red versions,
       and the negative control (c,c',c',c).
  c3 : Cor. C.3 and its converse on random words (products of two elementary moves).
"""
import random
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from R01_lib import is_red, seq_str, parse_seq, random_reduced_word, lehmer_word  # noqa
from alg import Alg  # noqa


def config_states(col, k, I, j, F):
    """source/target states (tokens) and perm (source pos -> target pos, 0-based)."""
    m = len(col)
    sigma = {s: s for s in range(1, m + 1)}
    for a, b in F:
        sigma[a], sigma[b] = b, a
    J = (set(I) - {j} - {b for a, b in F}) | {a for a, b in F}

    def order(Iset):
        keys = [(-(s) if s not in Iset else s, s) for s in range(1, m + 1)] + [(0, "W")]
        keys.sort()
        return [x[1] for x in keys]
    src = order(set(I))
    tgt = order(J)
    tok = lambda s: ("W", k) if s == "W" else col[s - 1]
    src_t = tuple(tok(s) for s in src)
    tgt_t = tuple(tok(s) for s in tgt)
    perm = []
    for s in src:
        image = "W" if s == "W" else sigma[s]
        perm.append(tgt.index(image))
    return src_t, tgt_t, perm


def elements_over_words(A, src, perm, rng, n_words):
    els = []
    words = [lehmer_word(perm)] + [random_reduced_word(perm, rng) for _ in range(n_words)]
    for w in words:
        els.append(A.word(seq_str(src), tuple(i + 1 for i in w)))
    distinct = []
    for e in els:
        if not any(e == d for d in distinct):
            distinct.append(e)
    return els[0], distinct, len(words)


def b7(rng):
    col = [1, 1, 2, 2, 1, 1, 2, 1]
    out = {}
    for F in ([(1, 8), (2, 5), (4, 7)], [(1, 8), (2, 6), (4, 7)]):
        I = {3, 5, 6, 7, 8}
        if F[1] == (2, 6):
            I = {3, 5, 6, 7, 8}
        src, tgt, perm = config_states(col, 1, I, 3, F)
        A = Alg(["A", 2], seq_str(src))
        basis, distinct, nw = elements_over_words(A, src, perm, rng, 80)
        print("b7: word 11221121, F=%s: %d reduced words -> %d distinct elements" % (F, nw, len(distinct)))
        for d in distinct[1:3]:
            print("     difference to the basis braid:", d - basis)
        out[tuple(F)] = len(distinct)
    return out[((1, 8), (2, 5), (4, 7))] > 1 and out[((1, 8), (2, 6), (4, 7))] == 1


def b5(rng):
    col = [1, 2, 3, 2, 1, 2, 2, 3]
    ok = True
    for k in (1, 2, 3):
        for I in ({4, 5, 6, 7, 8}, {5, 6, 7, 8}):
            src, tgt, perm = config_states(col, k, I, 7, [(1, 5), (2, 6), (3, 8)])
            A = Alg(["A", 3], seq_str(src))
            basis, distinct, nw = elements_over_words(A, src, perm, rng, 60)
            print("b5: word 12321223, red %d, I=%s: %d reduced words -> %d distinct element(s)"
                  % (k, sorted(I), nw, len(distinct)))
            ok &= len(distinct) == 1
    return ok


def reduced_words_4231():
    return [(1, 2, 3, 2, 1), (1, 3, 2, 1, 3), (3, 1, 2, 1, 3), (1, 3, 2, 3, 1), (3, 1, 2, 3, 1), (3, 2, 1, 2, 3)]


def b8():
    ok = True
    for ct, seq, expect_one in [(["A", 2], "1 2 1 1", True), (["A", 2], "1 1 2 1", True),
                                (["A", 2], "2 1 2 2", True), (["A", 2], "1 W1 1 1", True),
                                (["A", 2], "1 1 W1 1", True), (["A", 2], "1 2 2 1", False),
                                (["A", 3], "2 3 2 2", True), (["A", 2], "1 W1 W1 1", None)]:
        if "W1 W1" in seq:
            continue
        A = Alg(ct, seq)
        els = [A.word(seq, w) for w in reduced_words_4231()]
        distinct = []
        for e in els:
            if not any(e == d for d in distinct):
                distinct.append(e)
        print("b8: state %-9s : 6 reduced words of 4231 -> %d distinct" % (seq, len(distinct)))
        if expect_one is True:
            ok &= len(distinct) == 1
        elif expect_one is False:
            ok &= len(distinct) > 1
    return ok


def c3(rng, n=60):
    """products m_i(I) m_j(I-i) vs m_j(I) m_i(I-j)."""
    agree = differ = wrong = 0
    for _ in range(n):
        ct = rng.choice([["A", 2], ["A", 3]])
        nc = ct[1]
        m = rng.randint(3, 6)
        col = [rng.randint(1, nc) for _ in range(m)]
        k = rng.randint(1, nc)
        I = set(s for s in range(1, m + 1) if rng.random() < 0.6)
        if len(I) < 2:
            continue
        i, j = sorted(rng.sample(sorted(I), 2))
        adj = lambda a, b: abs(a - b) == 1
        has_bad = any(col[x - 1] == col[j - 1] and adj(col[j - 1], col[i - 1]) for x in range(1, i))

        def move(Iset, s):
            src, tgt, perm = config_states(col, k, Iset, s, [])
            return src, tuple(x + 1 for x in lehmer_word(perm))
        src, w1 = move(I, i)
        _, w2 = move(I - {i}, j)
        _, w3 = move(I, j)
        _, w4 = move(I - {j}, i)
        A = Alg(ct, seq_str(src))
        e1 = A.word(seq_str(src), w1 + w2)
        e2 = A.word(seq_str(src), w3 + w4)
        same = e1 == e2
        if same and not has_bad:
            agree += 1
        elif (not same) and has_bad:
            differ += 1
        else:
            wrong += 1
            print("  c3 unexpected", col, k, sorted(I), i, j, has_bad, same)
    print("c3: %d cases without bad (x,i,j): products equal; %d with: products differ; %d unexpected"
          % (agree, differ, wrong))
    return wrong == 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    rng = random.Random(3)
    ok = True
    if mode in ("b8", "all"):
        ok &= b8()
    if mode in ("c3", "all"):
        ok &= c3(rng)
    if mode in ("b5", "all"):
        ok &= b5(rng)
    if mode in ("b7", "all"):
        ok &= b7(rng)
    print("ALL OK" if ok else "FAILURES")
