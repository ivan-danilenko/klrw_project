"""
Type A (Theorem A of proofs/03_cyclic_combinatorics.md).  Pure Python.

    python3 proofs/checks/03_typeA.py [BOX]          (default BOX = 7)

1. Model check: for every k x (n-k) rectangle with 1 <= k, n-k <= BOX, the row-major word
   of the boxes, box (r, s) of colour k + s - r, is a minuscule chain omega_k -> lowest weight
   of sl_n (so its heap is the minuscule poset), and its heap order is the componentwise
   order on boxes  (r,s) <= (r',s')  iff  r <= r' and s <= s'.  Hence the order ideals are the
   Young diagrams in the rectangle and the word of lambda is the row-major word of lambda.
2. For every partition lambda in the BOX x BOX box: exhaustive search (03_lib.Word.search)
   decides cyclicity; compare with  lambda >= (4,4,4,3).  Also checks the two facts used in
   the proof: lambda not >= (4,4,4,3) => every colour occurs at most 3 times; and the
   explicit configuration of PROOFS.md Prop. 8.
3. All inclusion-minimal cyclic configurations in some diagrams, classified by shape.
4. addable_check: the claim in the proof of Lemma 3.1(d) (pairing = 1 iff the next box of the
   diagonal is addable); all_cyclic_brute: Prop. 3.3(a) addendum (unique cyclic configuration).
"""
import importlib, os, sys, itertools
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
L = importlib.import_module("03_lib")

ADJ = L.adjacency("Z")


def partitions_in_box(a, b):
    """Partitions with at most a rows and parts <= b."""
    def rec(i, mx):
        if i == a:
            yield ()
            return
        for p in range(mx, -1, -1):
            for rest in rec(i + 1, p):
                yield (p,) + rest
    for lam in rec(0, b):
        yield tuple(x for x in lam if x > 0)


def boxes(lam):
    return [(r, s) for r in range(1, len(lam) + 1) for s in range(1, lam[r - 1] + 1)]


def word_of(lam):
    B = boxes(lam)
    return B, [s - r for (r, s) in B]          # colour up to the constant k


def contains(lam, mu):
    return len(lam) >= len(mu) and all(lam[i] >= mu[i] for i in range(len(mu)))


def model_check(BOX):
    for k in range(1, BOX + 1):
        for nk in range(1, BOX + 1):
            n = k + nk
            lam = (nk,) * k
            B, _ = word_of(lam)
            word = [k + s - r for (r, s) in B]
            L.chain_weights("A", n - 1, k, word)           # raises unless a chain
            final = L.chain_weights("A", n - 1, k, word)[-1]
            assert all(final[i] != 1 for i in range(1, n)), "not the lowest weight"
            less = L.heap_order(word, L.adjacency("A", n - 1))
            for x, (r, s) in enumerate(B):
                for y, (r2, s2) in enumerate(B):
                    assert (y in less[x]) == ((r, s) != (r2, s2) and r <= r2 and s <= s2)
    print("model check: rectangles up to %dx%d are minuscule chains, heap = componentwise order" % (BOX, BOX))


def prop8_config(lam):
    """The configuration of PROOFS.md Prop. 8, as slot indices of the row-major word."""
    B, col = word_of(lam)
    ix = {b: i for i, b in enumerate(B)}
    j = ix[(3, 3)]
    F = tuple(sorted([(ix[(1, 1)], ix[(2, 2)]), (ix[(2, 1)], ix[(4, 3)]), (ix[(1, 2)], ix[(3, 4)])]))
    return j, F


def main(BOX):
    model_check(min(BOX, 6))
    Mi = importlib.import_module("03_minuscule")
    for n in range(2, 11):
        for k in range(1, n):
            Mi.weight_order_check("A", n - 1, k)
    print("weight order = reverse inclusion of ideals: sl_n, all omega_k, n <= 10")
    hexa = (4, 4, 4, 3)
    n = 0
    stats = Counter()
    for lam in partitions_in_box(BOX, BOX):
        B, col = word_of(lam)
        W = L.Word(col, ADJ)
        cyc = W.search() is not None
        exp = contains(lam, hexa)
        assert cyc == exp, (lam, cyc)
        if not exp:
            assert max(Counter(col).values(), default=0) <= 3, lam       # Lemma 3.2
        else:
            j, F = prop8_config(lam)
            assert W.is_cyclic_config(j, F), lam
        stats[cyc] += 1
        n += 1
    print("all %d partitions in the %dx%d box: cyclic <=> lambda contains (4,4,4,3)   (%d cyclic, %d not)" % (
        n, BOX, BOX, stats[True], stats[False]))


def describe_minimal(lam, show=False):
    B, col = word_of(lam)
    W = L.Word(col, ADJ)
    mins = W.minimal_cyclic()
    kinds = Counter()
    for j, F in sorted(mins):
        sh = L.shape(W, j, F)
        assert sh["chordless"]
        cc = sh["cycle_colours"]
        kinds[(sh["cycle_len"], sh["mover_dist"], sh["npairs"])] += 1
        if show:
            print("   mover %s, pairs %s; cycle length %d, mover at distance %d, colours along the cycle %s" % (
                B[j], [(B[a], B[b]) for a, b in F], sh["cycle_len"], sh["mover_dist"], cc))
    print("lambda = %s: %d minimal cyclic configurations; (cycle length, mover distance, #pairs): %s" % (
        lam, len(mins), dict(sorted(kinds.items()))))
    return mins


def all_cyclic_brute(lam):
    """Every cyclic configuration of lambda, by brute force over all movers and families."""
    B, col = word_of(lam)
    W = L.Word(col, ADJ)
    out = []
    for j in range(W.m):
        for F in L.matchings([p for p in W.pairs if j not in p]):
            if F and W.is_cyclic_config(j, F):
                out.append((B[j], [(B[a], B[b]) for a, b in F]))
    return out


def addable_check(BOX):
    """Lemma 3.1(c), induction step: for every diagram lambda in the box and every diagonal d,
    the pairing <mu_lambda, alpha_{k+d}^vee> = delta_{d,0} + a_{d-1} + a_{d+1} - 2 a_d lies in
    {-1, 0, 1}, and equals 1 iff the next box of D_d is addable to lambda."""
    cnt = 0
    for lam in partitions_in_box(BOX, BOX):
        box = set(boxes(lam))
        a = lambda d: sum(1 for (r, s) in box if s - r == d)
        for d in range(-BOX + 1, BOX):
            pairing = (d == 0) + a(d - 1) + a(d + 1) - 2 * a(d)
            i = a(d) + 1                                   # index of the next box of D_d
            y = (i, i + d) if d >= 0 else (i - d, i)
            r, s = y
            addable = (r <= BOX and s <= BOX and y not in box
                       and (s == 1 or (r, s - 1) in box) and (r == 1 or (r - 1, s) in box))
            assert pairing in (-1, 0, 1), (lam, d)
            assert (pairing == 1) == addable, (lam, d, pairing, addable)
            cnt += 1
    print("Lemma 3.1(c) step: pairing in {-1,0,1}, = 1 iff the next box is addable "
          "(%d diagram/diagonal pairs, %dx%d box, k = n - k = %d)" % (cnt, BOX, BOX, BOX))


if __name__ == "__main__":
    addable_check(6)
    for lam in [(4, 4, 4, 3), (5, 4, 4, 3), (4, 4, 4, 3, 1), (5, 5, 4, 3)]:
        allc = all_cyclic_brute(lam)
        assert len(allc) == 1 and allc[0][0] == (3, 3), (lam, allc)
    print("Prop. 3.3(a): (4,4,4,3), (5,4,4,3), (4,4,4,3,1), (5,5,4,3) have exactly one cyclic "
          "configuration (brute force over all configurations)")
    BOX = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    main(BOX)
    describe_minimal((4, 4, 4, 3), show=True)
    for lam in [(4, 4, 4, 4), (5, 4, 4, 3), (4, 4, 4, 3, 1), (5, 5, 4, 3), (5, 5, 5, 5), (5, 5, 5, 5, 5)]:
        describe_minimal(lam)
