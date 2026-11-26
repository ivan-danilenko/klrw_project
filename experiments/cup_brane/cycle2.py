"""
"Clean" cycle formula: tree terms plus, for configurations whose link graph has one cycle,

    sum over cycle links e = (x, y)  [b_x < b_y; the mover counts with b = j]
        sign(T_e) * mono(T_e) * ( t_{c_x c_y} * [dot on x] + t_{c_y c_x} * [dot on y] ) * beta

T_e = G - e, beta = the reduced exchange diagram drawn in canonical time order (right-hand
events from the inside out, left-hand events from the outside in).  Each link is of type R
or L, alternating around the cycle:

    type R:  dot on the right strand of x at the source (mover: at the source),
             dot on the right strand of y right after its own move;
    type L:  dot on the left strand of x at the target (mover: at the target),
             dot on the left strand of y right before its own move.

    sage -python cycle2.py pieces/FILE.json ...
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from fast import braid_for, exp_for
from formula import Stack
from events import simulate, canonical_orders
from decode import term_keys
from cycle import link_list, configurations, spanning_trees, cycle_order
from sage.rings.rational_field import QQ


def bkey(j, ev):
    return j if ev == "m" else ev[1]


def dot_insert(cube, V, word, tags, strand, where):
    """
    Element of ``word`` with a dot on ``strand`` (source slot).  ``where``:
    ("src",), ("tgt",), ("after", side, event) = right after the last letter of that event,
    ("before", side, event) = right before its first letter.
    """
    KLRW = cube.KLRW
    n = cube.n
    col = cube.strand_colour
    if where[0] == "src":
        idx = 0
    elif where[0] == "tgt":
        idx = len(word)
    else:
        start, end = tags[where[1], where[2]]
        idx = end if where[0] == "after" else start
    eps = tuple(1 if s + 1 in V else 0 for s in range(n))
    cur = [x if isinstance(x, tuple) else x + 1 for x in cube.labels(eps)]
    left_state = cube.state(eps)
    mid_state = left_state
    for letter in word[:idx]:
        cur[letter - 1], cur[letter] = cur[letter], cur[letter - 1]
        mid_state = mid_state.act_by_s(letter)
    prefix = KLRW.element_from_braid(word[:idx], left_state=left_state) if idx else KLRW.idempotent(left_state)
    suffix = KLRW.element_from_braid(word[idx:], left_state=mid_state) if idx < len(word) else KLRW.idempotent(mid_state)
    pos = cur.index(strand)
    rank = sum(1 for x in cur[: pos + 1] if not isinstance(x, tuple) and col[x - 1] == col[strand - 1])
    x = KLRW.base()("x_%d_%d" % (col[strand - 1], rank))
    return prefix * (x * suffix)


def cyclic_element(cube, stack, V, j, F, links, cache, variant):
    """Clean cyclic contribution of a configuration with exactly one cycle (without the factor kappa-sign,
    which is contained in stack.sign)."""
    KLRW = cube.KLRW
    D = KLRW.base()
    col = stack.col
    ro, lo = canonical_orders(j, F)
    tags = {}
    word = simulate(cube, V, j, F, ro, lo, None, tags)
    cyc = cycle_order(F, links)
    total = KLRW.zero()
    for subset, parent in spanning_trees(F, links):
        (extra,) = [i for i in range(len(links)) if i not in subset]
        mono = stack.monomial(parent)
        sign = stack.sign(V, j, F, parent)
        if mono not in cache:
            cache[mono] = D.monomial(*exp_for(KLRW, mono)) if mono else D.one()
        pos = cyc.index(extra)
        first = "R" if variant == "A" else "L"
        other = "L" if first == "R" else "R"
        side = first if pos % 2 == 0 else other
        x, y = links[extra][0], links[extra][1]
        if bkey(j, x) > bkey(j, y):
            x, y = y, x
        cx = col[j - 1] if x == "m" else col[x[0] - 1]
        cy = col[j - 1] if y == "m" else col[y[0] - 1]
        tx = D("t_%d_%d" % (cx, cy))
        ty = D("t_%d_%d" % (cy, cx))
        if side == "R":
            sx = j if x == "m" else x[1]          # right strand of x starts in slot b_x
            sy = y[1]
            dx = dot_insert(cube, V, word, tags, sx, ("src",))
            dy = dot_insert(cube, V, word, tags, sy, ("after", "R", y))
        else:
            sx = j if x == "m" else x[0]          # left strand of x starts in slot a_x
            sy = y[0]
            dx = dot_insert(cube, V, word, tags, sx, ("tgt",))
            dy = dot_insert(cube, V, word, tags, sy, ("before", "L", y))
        total += sign * cache[mono] * (tx * dx + ty * dy)
    return total


def compare(path, variant="A", verbose=True):
    data = json.load(open(path))
    ct, red, word = data["cartan_type"], data["red"], tuple(data["word"])
    P = [(red, word)]
    KLRW = make_algebra(ct, P); cube = Cube(KLRW, P, build=False); stack = Stack(ct, word)
    truth = {}
    for src, tgt, sigma, mono, dots, c in data["terms"]:
        truth.setdefault((frozenset(src), frozenset(tgt)), {})[
            (tuple(sigma), tuple(tuple(x) for x in mono), tuple(tuple(x) for x in dots))] = c
    vertices = [frozenset(v) for v in data["vertices"]]
    vset = set(vertices)
    cache = {}
    n_ok = n_bad = n_cyc = n_cyc_ok = 0
    for V in vertices:
        pred = {}
        cyc_targets = set()
        for J, sigma, mono, sign in stack.terms_from(V):
            if J in vset:
                d = pred.setdefault(J, {})
                d[(sigma, mono, ())] = d.get((sigma, mono, ()), 0) + sign
        for j in sorted(V):
            for F, links in configurations(stack, V, j, max_b1=1).items():
                if len(links) == len(F):
                    continue
                U = (V - {j} - {b for a, b in F}) | {a for a, b in F}
                if U not in vset:
                    continue
                el = cyclic_element(cube, stack, V, j, F, links, cache, variant)
                d = pred.setdefault(U, {})
                for k, c in term_keys(cube, V, U, el).items():
                    d[k] = d.get(k, 0) + c
                cyc_targets.add(U)
        for U in set(pred) | set(u for (v, u) in truth if v == V):
            p = {k: c for k, c in pred.get(U, {}).items() if c}
            t = truth.get((V, U), {})
            ok = (p == t)
            n_ok += ok; n_bad += (not ok)
            if U in cyc_targets:
                n_cyc += 1; n_cyc_ok += ok
            if not ok and verbose and n_bad <= 2:
                print("  MISMATCH V=%s U=%s (cyclic: %s)" % (sorted(V), sorted(U), U in cyc_targets))
                for k in sorted(set(p) | set(t), key=str):
                    if p.get(k, 0) != t.get(k, 0):
                        mv = {s + 1: tt for s, tt in enumerate(k[0]) if s + 1 != tt}
                        print("       pred %s true %s  %s dots %s sigma %s" % (p.get(k, 0), t.get(k, 0),
                              "*".join(n_ if pw == 1 else "%s^%d" % (n_, pw) for n_, pw in k[1]), k[2], mv))
    print("%-52s [%s]: components ok %d, wrong %d; cyclic components ok %d of %d" % (
        os.path.basename(path).replace("__K_.json", ""), variant, n_ok, n_bad, n_cyc_ok, n_cyc))
    return n_bad == 0


if __name__ == "__main__":
    for p in sys.argv[1:]:
        for variant in ("A", "B"):
            compare(p, variant=variant, verbose=False)
