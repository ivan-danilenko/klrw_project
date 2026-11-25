"""
Targeted d^2 check: vertices in which a colour with >= 4 slots d1<d2<d3<d4 has
d1, d2 on the left and d3, d4 on the right (so that two exchange pairs of the
same colour are available), all other slots random.

    sage -python targeted.py TYPE RANK RED [min|max|WORD] SAMPLES [LENGTH] [SEED]
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_verify import *

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]; red = int(sys.argv[3]); spec = sys.argv[4]
    samples = int(sys.argv[5])
    length = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "full" else None
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    word = minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length) if spec in ("min", "max") else tuple(int(c) for c in spec)
    m = len(word)
    print("type", ct, "red", red, "word", "".join(map(str, word)), "strands", m, flush=True)
    P = [(red, word)]
    KLRW = make_algebra(ct, P); cube = Cube(KLRW, P, build=False); stack = Stack(ct, word)
    quads = []
    for (p, c), sl in stack.same.items():
        for k in range(len(sl) - 3):
            quads.append((c, sl[k:k + 4]))
    print("quadruples of equal colour:", quads, flush=True)
    rng = random.Random(seed); cache = {}; fails = 0; t0 = time.time()
    for k in range(samples):
        c, (d1, d2, d3, d4) = rng.choice(quads)
        I = set(s for s in range(1, m + 1) if rng.random() < 0.5)
        I -= {d1, d2}; I |= {d3, d4}
        # make sure the adjacent-coloured slots between d2 and d3 are on the right (movers)
        for s in range(d2 + 1, d3):
            if stack.adjacent(s, d2):
                I.add(s)
        I = frozenset(I)
        t1 = time.time()
        n_prod, n_targets, bad, order = check_vertex(cube, stack, I, cache)
        fails += bool(bad)
        print("  sample {:3d}: colour {} quad {}, |I| = {:2d}, {:6d} products, max order {}, ok: {}  [{:.0f}s]".format(
            k, c, (d1, d2, d3, d4), len(I), n_prod, order, not bad, time.time() - t1), flush=True)
        if bad:
            print("      I =", sorted(I), " failing targets:", [(sorted(I - K), sorted(K - I)) for K in bad][:4], flush=True)
    print("RESULT {} samples, {} failures [{:.0f}s]".format(samples, fails, time.time() - t0), flush=True)
