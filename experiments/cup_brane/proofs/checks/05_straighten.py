"""
Check of the Straightening Lemma (Lemma 1.2 of proofs/05_main_sliding.md).

For a word D in the crossings, starting at a state s (black strands and one red strand), let n(A, B)
be the number of crossings of the strands A and B.  If

  (S1) n(A, B) >= 2  implies: A, B black of different non-adjacent colours, or one of them is red and
       the other is black of a colour different from the red one;
  (S2) there are no three strands A, B, C with n(A,B), n(B,C), n(A,C) >= 1 such that A, C are black of
       the same colour and B is black of an adjacent colour, or B is red of that same colour,

then the element of D equals the element of every reduced word of the permutation of D.

Random test: random states and random words; whenever (S1), (S2) hold the equality is checked against
two random reduced words.  For comparison the script also counts how often the equality fails when
(S1) or (S2) is violated (so the hypotheses are not vacuous).

Also checked (Lemmas 1.3 and 1.4), always assuming (S2):
  * (vanishing) if every pair with n >= 2 is as in (S1) or consists of two black strands of the same
    colour, and there is a pair of the second kind, then the element of D is 0;
  * (bigons) if no two strands of the same colour cross twice, and no strand Z of a pair {A, B} with
    n(A,B) >= 2 of "adjacent type" (black of adjacent colours, or red and black of the red colour)
    crosses a strand of its own colour, then
        element of D  =  prod_{such pairs} Q_{AB}^{floor(n(A,B)/2)}  *  (reduced diagram),
    Q_{AB} = t_{c(A)c(B)} x_A + t_{c(B)c(A)} x_B  resp.  u_k x_A,  the dots at the source.

    sage -python 05_straighten.py [N_SAMPLES] [SEED]
"""
import sys, random
from collections import Counter
from alg import Alg
from degree_model import edges as dynkin_edges


def hypotheses(ct, seq, word):
    E = dynkin_edges(ct)
    cur = list(range(len(seq)))
    cnt = Counter()
    for i in word:
        cnt[frozenset((cur[i - 1], cur[i]))] += 1
        cur[i - 1], cur[i] = cur[i], cur[i - 1]
    red = lambda x: seq[x][0] == "W"
    col = lambda x: int(seq[x][1:]) if red(x) else int(seq[x])
    ok1 = True
    for pair, n in cnt.items():
        if n >= 2:
            x, y = tuple(pair)
            if red(x) or red(y):
                if col(x) == col(y):
                    ok1 = False
            elif col(x) == col(y) or frozenset((col(x), col(y))) in E:
                ok1 = False
    ok2 = True
    n = len(seq)
    for A in range(n):
        for C in range(A + 1, n):
            if red(A) or red(C) or col(A) != col(C) or frozenset((A, C)) not in cnt:
                continue
            for B in range(n):
                if B in (A, C) or frozenset((A, B)) not in cnt or frozenset((B, C)) not in cnt:
                    continue
                if red(B):
                    if col(B) == col(A):
                        ok2 = False
                elif frozenset((col(A), col(B))) in E:
                    ok2 = False
    return ok1, ok2, cur


def classify(ct, seq, word):
    """(S2), and the pairs crossing at least twice sorted into: harmless (as in (S1)), adjacent type,
    same colour.  Also: does some strand of an adjacent-type pair cross a strand of its own colour?"""
    E = dynkin_edges(ct)
    cur = list(range(len(seq)))
    cnt = Counter()
    for i in word:
        cnt[frozenset((cur[i - 1], cur[i]))] += 1
        cur[i - 1], cur[i] = cur[i], cur[i - 1]
    red = lambda x: seq[x][0] == "W"
    col = lambda x: int(seq[x][1:]) if red(x) else int(seq[x])
    adjacent_type, same = [], []
    for pair, n in cnt.items():
        if n >= 2:
            x, y = sorted(pair)
            if red(x) or red(y):
                if col(x) == col(y):
                    adjacent_type.append((x, y, n))
            elif col(x) == col(y):
                same.append((x, y, n))
            elif frozenset((col(x), col(y))) in E:
                adjacent_type.append((x, y, n))
    free = True
    for x, y, n in adjacent_type:
        for z in (x, y):
            if red(z):
                continue
            for pair in cnt:
                if z in pair:
                    (o,) = pair - {z}
                    if not red(o) and col(o) == col(z):
                        free = False
    return adjacent_type, same, free


def random_reduced(n, final, rng):
    rank = {s: i for i, s in enumerate(final)}
    cur = list(range(n))
    word = []
    while True:
        desc = [i for i in range(n - 1) if rank[cur[i]] > rank[cur[i + 1]]]
        if not desc:
            return tuple(word)
        i = rng.choice(desc)
        cur[i], cur[i + 1] = cur[i + 1], cur[i]
        word.append(i + 1)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    good = good_nonreduced = 0
    viol = viol_fail = 0
    n_van = n_big = 0
    algs = {}
    for _ in range(N):
        ct = rng.choice([["A", 3], ["A", 4], ["D", 4]])
        rank = ct[1]
        n = rng.randint(3, 6)
        seq = [str(rng.randint(1, rank)) for _ in range(n - 1)]
        seq.insert(rng.randint(0, n - 1), "W%d" % rng.randint(1, rank))
        word = tuple(rng.randint(1, n - 1) for _ in range(rng.randint(2, 9)))
        ok1, ok2, final = hypotheses(ct, seq, word)
        key = (tuple(ct), tuple(sorted(seq)))
        if key not in algs:
            algs[key] = Alg(ct, seq)
        A = algs[key]
        el = A.word(seq, word)
        red_words = [random_reduced(n, final, rng) for _ in range(2)]
        equal = all(A.word(seq, rw) == el for rw in red_words)
        if ok1 and ok2:
            assert equal, (ct, seq, word)
            good += 1
            good_nonreduced += len(red_words[0]) < len(word)
        else:
            viol += 1
            viol_fail += not equal
        if ok2 and not ok1:
            adjacent_type, same, free = classify(ct, seq, word)
            if same and not adjacent_type:
                assert el.is_zero(), (ct, seq, word)                       # Lemma 1.4
                n_van += 1
            if adjacent_type and not same and free:                        # Lemma 1.3
                P = A.e(seq)
                for x, y, nn in adjacent_type:
                    if seq[x][0] == "W" or seq[y][0] == "W":
                        b = y if seq[x][0] == "W" else x
                        Q = A.u(int(seq[b])) * A.dot(seq, b + 1)
                    else:
                        cx, cy = int(seq[x]), int(seq[y])
                        Q = A.t(cx, cy) * A.dot(seq, x + 1) + A.t(cy, cx) * A.dot(seq, y + 1)
                    for _ in range(nn // 2):
                        P = P * Q
                assert el == P * A.word(seq, red_words[0]), (ct, seq, word)
                n_big += 1
    print("straightening lemma: %d words satisfy (S1),(S2) (%d of them non-reduced): all equal to the reduced diagram"
          % (good, good_nonreduced))
    print("                     %d words violate (S1) or (S2); for %d of them the equality fails" % (viol, viol_fail))
    print("vanishing lemma 1.4: %d words, all zero;   bigon lemma 1.3: %d words, all equal to the prediction" % (n_van, n_big))
