"""Search for a failing vertex I of small size:  sage -python find_small_failure.py TYPE RANK RED SPEC MAXSIZE TRIES [SEED]"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_verify import *

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]; red = int(sys.argv[3]); spec = sys.argv[4]
    maxsize = int(sys.argv[5]); tries = int(sys.argv[6]); seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    word = minuscule_word(ct, red, {"min": min, "max": max}[spec]) if spec in ("min", "max") else tuple(int(c) for c in spec)
    m = len(word)
    P = [(red, word)]
    KLRW = make_algebra(ct, P); cube = Cube(KLRW, P, build=False); stack = Stack(ct, word)
    print("word", "".join(map(str, word)), " slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(word)), flush=True)
    rng = random.Random(seed); cache = {}
    best = None
    for k in range(tries):
        size = rng.randint(3, maxsize)
        # bias towards late slots (so that exchange pairs are available)
        I = frozenset(rng.sample(range(m // 3, m + 1), size))
        n_prod, n_targets, bad, order = check_vertex(cube, stack, I, cache)
        if bad:
            print("FAIL |I|=%d I=%s  targets (out, in): %s" % (len(I), sorted(I), [(sorted(I - K), sorted(K - I)) for K in bad][:3]), flush=True)
            if best is None or len(I) < len(best):
                best = I
    print("smallest failing I found:", sorted(best) if best else None)
