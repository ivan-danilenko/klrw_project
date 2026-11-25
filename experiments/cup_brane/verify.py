"""
Build the differential of a stack directly from the closed formula
(``formula.Stack``) and check d^2 = 0 exactly in the KLRW algebra.

    sage -python verify.py TYPE RANK RED [min|max|WORD|all] [LENGTH]
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from cupbrane import _to_csc
from fast import braid_for, exp_for
from formula import Stack


def formula_differential(cube, stack):
    KLRW = cube.KLRW
    D = KLRW.base()
    entries = {h: defaultdict(lambda: KLRW.zero()) for h in range(1, cube.n + 1)}
    count = 0
    mono_cache = {}
    for (I, J, sigma, mono), sign in stack.all_terms().items():
        eps_I, eps_J, braid = braid_for(cube, I, J, sigma)
        if mono not in mono_cache:
            mono_cache[mono] = D.monomial(*exp_for(KLRW, mono)) if mono else D.one()
        h, col = cube.index[eps_I]
        _, row = cube.index[eps_J]
        entries[h][row, col] += KLRW.term(braid, D(sign) * mono_cache[mono])
        count += 1
    return {h: dict(e) for h, e in entries.items()}, count


def verify(ct, red, word, verbose=True):
    t0 = time.time()
    P = [(red, tuple(word))]
    KLRW = make_algebra(ct, P)
    cube = Cube(KLRW, P)
    stack = Stack(ct, word)
    diff, count = formula_differential(cube, stack)
    # homogeneity
    for h, e in diff.items():
        for (row, col), entry in e.items():
            deg = entry.degree(check_if_homogeneous=True)
            assert deg == cube.projectives[h - 1][row].equivariant_degree - cube.projectives[h][col].equivariant_degree
    d_csc = _to_csc(KLRW, cube.projectives, diff)
    t1 = time.time()
    ok = check_square_zero(KLRW, cube.projectives, d_csc)
    if verbose:
        print("{} red {} word {}: {} strands, {} terms, homogeneous, d^2 = 0: {}   [build {:.0f}s, square {:.0f}s]".format(
            "".join(map(str, ct)), red, "".join(map(str, word)), len(word), count, ok, t1 - t0, time.time() - t1), flush=True)
    return ok


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    red = int(sys.argv[3])
    spec = sys.argv[4] if len(sys.argv) > 4 else "min"
    length = int(sys.argv[5]) if len(sys.argv) > 5 else None
    if spec == "all":
        words = all_minuscule_words(ct, red, length=length)
    elif spec in ("min", "max"):
        words = [minuscule_word(ct, red, {"min": min, "max": max}[spec], length=length)]
    else:
        words = [tuple(int(c) for c in spec)]
    results = [verify(ct, red, w) for w in words]
    print("ALL OK" if all(results) else "FAILURES: {}".format(results.count(False)), "({} words)".format(len(words)))
