"""
Check (d^2)(e_I) = 0 for randomly chosen vertices I of a (possibly huge) stack,
using only the closed formula.  The full cube is never built.

    sage -python sample_verify.py TYPE RANK RED [min|max|WORD] SAMPLES [LENGTH] [SEED]
"""
import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from fast import braid_for, exp_for
from formula import Stack


def element(cube, stack, I, term, cache):
    J, sigma, mono, sign = term
    KLRW = cube.KLRW
    D = KLRW.base()
    eps_I, eps_J, braid = braid_for(cube, I, J, sigma)
    if mono not in cache:
        cache[mono] = D.monomial(*exp_for(KLRW, mono)) if mono else D.one()
    return KLRW.term(braid, D(sign) * cache[mono])


def check_vertex(cube, stack, I, cache):
    KLRW = cube.KLRW
    total = defaultdict(lambda: KLRW.zero())
    n_prod = 0
    orders = set()
    for t1 in stack.terms_from(I):
        x = element(cube, stack, I, t1, cache)
        J = t1[0]
        for t2 in stack.terms_from(J):
            y = element(cube, stack, J, t2, cache)
            total[t2[0]] += x * y
            n_prod += 1
            orders.add(len(t1[2]) // 2 + len(t2[2]) // 2)
    bad = [K for K, v in total.items() if not v.is_zero()]
    return n_prod, len(total), bad, max(orders) if orders else 0


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    spec = sys.argv[4]
    samples = int(sys.argv[5])
    length = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "full" else None
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    word = minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length) if spec in ("min", "max") else tuple(int(c) for c in spec)
    m = len(word)
    print("type", ct, "red", red, "word", "".join(map(str, word)), "strands", m, flush=True)
    P = [(red, word)]
    KLRW = make_algebra(ct, P)
    cube = Cube(KLRW, P, build=False)
    stack = Stack(ct, word)
    rng = random.Random(seed)
    cache = {}
    failures = 0
    t0 = time.time()
    for k in range(samples):
        size = rng.randint(2, m)
        I = frozenset(rng.sample(range(1, m + 1), size))
        t1 = time.time()
        n_prod, n_targets, bad, order = check_vertex(cube, stack, I, cache)
        failures += bool(bad)
        print("  sample {:3d}: |I| = {:2d}, {:6d} products, {:5d} targets, max order {}, d^2 e_I = 0: {}  [{:.0f}s]".format(
            k, size, n_prod, n_targets, order, not bad, time.time() - t1), flush=True)
    print("RESULT {} {} red {}: {} samples, {} failures  [{:.0f}s]".format(
        "".join(map(str, ct)), "".join(map(str, word)), red, samples, failures, time.time() - t0), flush=True)
