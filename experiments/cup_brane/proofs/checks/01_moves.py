"""
01_moves.py   (sage -python proofs/checks/01_moves.py [N])

Section C of 01_foundations.md: elementary moves and products of two of them.
For random words (A_2, A_3, D_4; repeated colours allowed) and random vertices I:
  (C1) m_j(I) has exactly ONE reduced word (all reduced words of its position permutation
       are enumerated);
  (C2) for i < j in I the words  m_i(I) m_j(I-i)  and  m_j(I) m_i(I-j)  are reduced and
       represent the same bijection; the triples of pairwise crossing strands are exactly
       {x, i, j} (x < i) and {red, i, j}; the middle strand is always i;
  (C3) in the algebra:  m_i(I) m_j(I-i) == m_j(I) m_i(I-j)  iff no triple {x, i, j} is bad
       (c(x) = c(j) adjacent to c(i)).  (Only "if" is a theorem; "only if" is recorded.)
"""
import itertools
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from cupbrane import make_algebra, Cube          # noqa: E402
from klrw.stable_envelopes import braid_from_marked_state   # noqa: E402
from degree_model import edges                   # noqa: E402


def labels(m, I):
    return [s for s in range(m, 0, -1) if s not in I] + ["W"] + sorted(I)


def word_between(src, tgt):
    """reduced word (Lehmer cocode, as klrw) from the label list src to tgt (same labels)."""
    return tuple(braid_from_marked_state(list(src), list(tgt)))


def all_reduced_words(perm, limit=5000):
    out = []

    def rec(cur, w):
        if len(out) >= limit:
            return
        desc = [p for p in range(len(cur) - 1) if cur[p] > cur[p + 1]]
        if not desc:
            out.append(tuple(w))
            return
        for p in desc:
            c = list(cur)
            c[p], c[p + 1] = c[p + 1], c[p]
            rec(c, w + [p + 1])
    rec(list(perm), [])
    return set(out)


def crossings(src, word):
    lab = list(src)
    out = []
    for i in word:
        out.append(frozenset((lab[i - 1], lab[i])))
        lab[i - 1], lab[i] = lab[i], lab[i - 1]
    return out, lab


def main(N=60, seed=5):
    rng = random.Random(seed)
    stats = dict(C1=0, C2=0, equal=0, differ=0, bad_and_differ=0, nobad_and_equal=0, mismatch=0)
    cache = {}
    for _ in range(N):
        ct = rng.choice([["A", 2], ["A", 3], ["D", 4]])
        E = edges(ct)
        m = rng.randint(3, 7)
        word = tuple(rng.randint(1, ct[1]) for _ in range(m))
        red = rng.randint(1, ct[1])
        key = (tuple(ct), red, word)
        if key not in cache:
            KLRW = make_algebra(ct, [(red, word)])
            cache[key] = (KLRW, Cube(KLRW, [(red, word)], build=False))
        KLRW, cube = cache[key]
        I = set(s for s in range(1, m + 1) if rng.random() < 0.6)
        if len(I) < 2:
            continue
        i, j = sorted(rng.sample(sorted(I), 2))
        # (C1)
        for jj, II in ((j, I), (i, I)):
            src, tgt = labels(m, II), labels(m, II - {jj})
            perm = [tgt.index(x) for x in src]
            rws = all_reduced_words(perm)
            assert len(rws) == 1, (word, II, jj, rws)
            stats["C1"] += 1
        # (C2)
        L0, L1, L2, L1b = labels(m, I), labels(m, I - {i}), labels(m, I - {i, j}), labels(m, I - {j})
        w12 = word_between(L0, L1) + word_between(L1, L2)
        w21 = word_between(L0, L1b) + word_between(L1b, L2)
        c12, end12 = crossings(L0, w12)
        c21, end21 = crossings(L0, w21)
        assert end12 == end21 == L2
        assert len(set(c12)) == len(c12) and len(set(c21)) == len(c21), "not reduced"
        assert set(c12) == set(c21)
        pairs = set(c12)
        strands = L0
        triples = [t for t in itertools.combinations(strands, 3)
                   if all(frozenset(p) in pairs for p in itertools.combinations(t, 2))]
        expected = {frozenset(("W", i, j))} | {frozenset((x, i, j)) for x in range(1, i)}
        assert {frozenset(t) for t in triples} == expected, (word, I, i, j, triples)
        for t in triples:
            mid = sorted(t, key=lambda x: L0.index(x))[1]
            assert mid == i
        stats["C2"] += 1
        # (C3)
        col = lambda s: word[s - 1]
        bad = [x for x in range(1, i) if col(x) == col(j) and frozenset((col(x), col(i))) in E]
        eps = lambda S: tuple(1 if s + 1 in S else 0 for s in range(m))
        e12 = KLRW.element_from_braid(w12, left_state=cube.state(eps(I)))
        e21 = KLRW.element_from_braid(w21, left_state=cube.state(eps(I)))
        eq = (e12 == e21)
        stats["equal" if eq else "differ"] += 1
        if bad and not eq:
            stats["bad_and_differ"] += 1
        elif not bad and eq:
            stats["nobad_and_equal"] += 1
        else:
            stats["mismatch"] += 1
            print("   bad triples", bad, "but equal:", eq, word, sorted(I), i, j)
    print("elementary moves:", stats)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
