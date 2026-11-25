"""Diagnose a failing vertex of sample_verify: print the non-zero components of d^2 e_I."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_verify import *

if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]; red = int(sys.argv[3]); spec = sys.argv[4]
    length = int(sys.argv[5]) if sys.argv[5] != "full" else None
    seed = int(sys.argv[6]); which = int(sys.argv[7])
    word = minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length)
    m = len(word)
    P = [(red, word)]
    KLRW = make_algebra(ct, P); cube = Cube(KLRW, P, build=False); stack = Stack(ct, word)
    rng = random.Random(seed)
    for k in range(which + 1):
        size = rng.randint(2, m); I = frozenset(rng.sample(range(1, m + 1), size))
    print("word", word)
    print("slots:", " ".join("%d:c%d" % (s + 1, c) for s, c in enumerate(word)))
    print("I =", sorted(I), " not in I:", sorted(set(range(1, m + 1)) - I))
    total = defaultdict(lambda: KLRW.zero()); contrib = defaultdict(list)
    cache = {}
    for t1 in stack.terms_from(I):
        x = element(cube, stack, I, t1, cache)
        for t2 in stack.terms_from(t1[0]):
            y = element(cube, stack, t1[0], t2, cache)
            total[t2[0]] += x * y
            contrib[t2[0]].append((t1, t2))
    for K, v in total.items():
        if v.is_zero():
            continue
        print("K =", sorted(K), " out", sorted(I - K), " in", sorted(K - I))
        eps_I = tuple(1 if s + 1 in I else 0 for s in range(m))
        for braid, coeff in v:
            arrive = cube.strand_map(eps_I, braid)
            eps_K = tuple(1 if s + 1 in K else 0 for s in range(m))
            tgt = cube.labels(eps_K)
            sigma = {a + 1: t + 1 for a, t in zip(arrive, tgt) if not isinstance(a, tuple)}
            moved = {s: t for s, t in sigma.items() if s != t}
            print("    residual:", coeff, " sigma moves", moved, " len", len(braid.word()))
        for t1, t2 in contrib[K]:
            mv1 = {s + 1: t for s, t in enumerate(t1[1]) if s + 1 != t}
            mv2 = {s + 1: t for s, t in enumerate(t2[1]) if s + 1 != t}
            print("      via J out %s in %s sigma %s %s %+d | then out %s in %s sigma %s %s %+d" % (
                sorted(I - t1[0]), sorted(t1[0] - I), mv1, "*".join(n for n, p in t1[2] for _ in range(p)) or "1", t1[3],
                sorted(t1[0] - K), sorted(K - t1[0]), mv2, "*".join(n for n, p in t2[2] for _ in range(p)) or "1", t2[3]))
