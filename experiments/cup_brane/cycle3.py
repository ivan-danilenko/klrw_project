"""
Cycle formula, version 3 ("kiss" rule).

For a configuration with one cycle in its link graph, and a cycle link e = (x, y)
(b_x < b_y, the mover counting with b = j), of type R or L:

  * take the canonical time order (right-hand events by increasing b, left-hand events by
    decreasing b) and exchange the order of x and y on that side only (x right after y on the
    right for type R; y right after x on the left for type L), keeping every other link in order;
  * in the resulting diagram the two strands cross twice; delete both crossings.  The result is
    a drawing of the reduced exchange diagram in which the two strands touch twice;
  * the strand that moves second carries its dot at the first touching point,
    the strand that moves first at the second touching point.

    E_cyc = sum_e sign(T_e) mono(T_e) ( t_{c_x c_y} [dot on x] + t_{c_y c_x} [dot on y] ),   T_e = G - e,

with the types alternating around the cycle.

    sage -python cycle3.py pieces/FILE.json ...
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from fast import exp_for
from formula import Stack
from events import simulate, canonical_orders
from decode import term_keys
from cycle import configurations, spanning_trees, cycle_order


def bkey(j, ev):
    return j if ev == "m" else ev[1]


def consistent(order, constraints):
    pos = {ev: i for i, ev in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in constraints)


def reordered(order, first, second, constraints):
    """Linear order respecting ``constraints`` (pairs (a, b): a before b), as close as possible
    to ``order`` (greedy topological sort with ``order`` as tie-break); None if impossible."""
    rank = {ev: i for i, ev in enumerate(order)}
    remaining = set(order)
    out = []
    while remaining:
        free = [ev for ev in remaining if not any(b == ev and a in remaining for a, b in constraints)]
        if not free:
            return None
        ev = min(free, key=lambda e: rank[e])
        out.append(ev)
        remaining.discard(ev)
    return out


def strand_word_trace(cube, V, word):
    n = cube.n
    eps = tuple(1 if s + 1 in V else 0 for s in range(n))
    cur = [x if isinstance(x, tuple) else x + 1 for x in cube.labels(eps)]
    crossings = []
    for letter in word:
        crossings.append((cur[letter - 1], cur[letter]))
        cur[letter - 1], cur[letter] = cur[letter], cur[letter - 1]
    return crossings


def element_with_dot(cube, V, word, idx, strand):
    """Diagram ``word`` from vertex V with a dot on the strand starting in slot ``strand``,
    inserted after the first ``idx`` letters."""
    KLRW = cube.KLRW
    n = cube.n
    col = cube.strand_colour
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


def kiss_elements(cube, stack, V, j, F, links, link_index, side):
    """(D_x, D_y) for the cycle link ``links[link_index]`` drawn as a bigon on ``side``; None if not realisable."""
    rel = [(x, y) if bkey(j, x) < bkey(j, y) else (y, x) for x, y, _, _ in links]
    x, y = rel[link_index]
    ro, lo = canonical_orders(j, F)
    other = [r for i, r in enumerate(rel) if i != link_index]
    if side == "R":
        new_ro = reordered(ro, x, y, other + [(y, x)])
        if new_ro is None:
            return None
        ro = new_ro
        lo_c = [(b, a) for a, b in rel]
        assert consistent(lo, lo_c)
        sx = j if x == "m" else x[1]
        sy = y[1]
        first, second = sy, sx          # y moves first on the right
    else:
        lo_other = [(b, a) for a, b in other]
        new_lo = reordered(lo, y, x, lo_other + [(x, y)])
        if new_lo is None:
            return None
        lo = new_lo
        sx = j if x == "m" else x[0]
        sy = y[0]
        first, second = sx, sy          # x moves first on the left
    word = simulate(cube, V, j, F, ro, lo)
    crossings = strand_word_trace(cube, V, word)
    hits = [i for i, (a, b) in enumerate(crossings) if {a, b} == {sx, sy}]
    if len(hits) != 2:
        return None
    t1, t2 = hits
    reduced = word[:t1] + word[t1 + 1:t2] + word[t2 + 1:]
    d_second = element_with_dot(cube, V, reduced, t1, second)
    d_first = element_with_dot(cube, V, reduced, t2 - 1, first)
    return (d_first, d_second) if first == sx else (d_second, d_first)


def role_types(j, links, cyc, first):
    """Types of the cycle links: they change at a node that is a source or a sink of the cycle
    for the order by b, and persist at a node lying between its two neighbours."""
    flip = {"R": "L", "L": "R"}
    types = [first]
    n = len(cyc)
    for i in range(n - 1):
        e1, e2 = links[cyc[i]], links[cyc[i + 1]]
        (node,) = set(e1[:2]) & set(e2[:2])
        o1 = [z for z in e1[:2] if z != node][0]
        o2 = [z for z in e2[:2] if z != node][0]
        pure = (bkey(j, o1) > bkey(j, node)) == (bkey(j, o2) > bkey(j, node))
        types.append(flip[types[-1]] if pure else types[-1])
    return types


def cyclic_element(cube, stack, V, j, F, links, cache, variant):
    KLRW = cube.KLRW
    D = KLRW.base()
    col = stack.col
    cyc = cycle_order(F, links)
    total = KLRW.zero()
    canon = [None]
    for subset, parent in spanning_trees(F, links):
        (extra,) = [i for i in range(len(links)) if i not in subset]
        mono = stack.monomial(parent)
        sign = stack.sign(V, j, F, parent)
        if mono not in cache:
            cache[mono] = D.monomial(*exp_for(KLRW, mono)) if mono else D.one()
        pos = cyc.index(extra)
        if variant in ("A", "B"):
            first = "R" if variant == "A" else "L"
            other = "L" if first == "R" else "R"
            sides = [first if pos % 2 == 0 else other]
        elif variant in ("roleR", "roleL"):
            sides = [role_types(j, links, cyc, "R" if variant == "roleR" else "L")[pos]]
        else:
            sides = ["R", "L"]
        x, y = links[extra][0], links[extra][1]
        if bkey(j, x) > bkey(j, y):
            x, y = y, x
        cx = col[j - 1] if x == "m" else col[x[0] - 1]
        cy = col[j - 1] if y == "m" else col[y[0] - 1]
        tx = D("t_%d_%d" % (cx, cy))
        ty = D("t_%d_%d" % (cy, cx))
        if variant in ("endL", "endR", "mix"):
            # types are chosen separately for the two ends of the link:
            # at a source/sink of the cycle the two links get different sides,
            # at an in-between node each link contributes on both sides.
            both = {side: kiss_elements(cube, stack, V, j, F, links, extra, side) for side in "RL"}
            if both["R"] is None or both["L"] is None:
                raise RuntimeError("bigon not realisable: %s %s" % (F, links[extra][:2]))
            for which, node, tcoef in ((0, x, tx), (1, y, ty)):
                # the other cycle link at this node
                nb = [i for i in cyc if i != extra and node in links[i][:2]]
                (other_link,) = nb
                o1 = [z for z in links[extra][:2] if z != node][0]
                o2 = [z for z in links[other_link][:2] if z != node][0]
                pure = (bkey(j, o1) > bkey(j, node)) == (bkey(j, o2) > bkey(j, node))
                if pure:
                    # the link that comes first in the cyclic order gets the first side
                    first_side = "L" if variant == "endL" else "R"
                    second_side = "R" if first_side == "L" else "L"
                    cyc_pos = {i: k for k, i in enumerate(cyc)}
                    # compare positions of the two links as seen from this node, using the smaller partner first
                    mine_first = bkey(j, o1) < bkey(j, o2) or (bkey(j, o1) == bkey(j, o2) and cyc_pos[extra] < cyc_pos[other_link])
                    side = first_side if mine_first else second_side
                    total += 2 * sign * cache[mono] * tcoef * both[side][which]
                elif variant != "mix":
                    total += 2 * sign * cache[mono] * tcoef * (both["L"][which] + both["R"][which])
                else:
                    # in-between node: dots in the canonical drawing, right before / right after the
                    # node's own move on each side (so every same-colour crossing is counted once)
                    if canon[0] is None:
                        ro_c, lo_c = canonical_orders(j, F)
                        sp = {}
                        w_c = simulate(cube, V, j, F, ro_c, lo_c, None, sp)
                        canon[0] = (w_c, sp)
                    w_c, sp = canon[0]
                    is_y = bkey(j, o1) < bkey(j, node)      # node has the larger b in this link
                    a_n, b_n = node
                    # as y: left strand before its move, right strand after its move;
                    # as x: left strand after its move, right strand before its move
                    l_idx = sp["L", node][0] if is_y else sp["L", node][1]
                    r_idx = sp["R", node][1] if is_y else sp["R", node][0]
                    total += 2 * sign * cache[mono] * tcoef * (
                        element_with_dot(cube, V, w_c, l_idx, a_n) + element_with_dot(cube, V, w_c, r_idx, b_n))
            continue
        for side in sides:
            res = kiss_elements(cube, stack, V, j, F, links, extra, side)
            if res is None:
                raise RuntimeError("bigon not realisable: %s %s %s" % (F, links[extra][:2], side))
            dx, dy = res
            total += (2 // len(sides)) * sign * cache[mono] * (tx * dx + ty * dy)
    return total      # carries a factor 2


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
    failed = None
    for V in vertices:
        pred = {}
        cyc_targets = set()
        for J, sigma, mono, sign in stack.terms_from(V):
            if J in vset:
                d = pred.setdefault(J, {})
                d[(sigma, mono, ())] = d.get((sigma, mono, ()), 0) + 2 * sign
        for j in sorted(V):
            for F, links in configurations(stack, V, j, max_b1=1).items():
                if len(links) == len(F):
                    continue
                U = (V - {j} - {b for a, b in F}) | {a for a, b in F}
                if U not in vset:
                    continue
                try:
                    el = cyclic_element(cube, stack, V, j, F, links, cache, variant)
                except RuntimeError as err:
                    failed = str(err)
                    continue
                d = pred.setdefault(U, {})
                for k, c in term_keys(cube, V, U, el).items():
                    d[k] = d.get(k, 0) + c
                cyc_targets.add(U)
        for U in set(pred) | set(u for (v, u) in truth if v == V):
            p = {k: c for k, c in pred.get(U, {}).items() if c}
            t = {k: 2 * c for k, c in truth.get((V, U), {}).items()}
            ok = (p == t)
            n_ok += ok; n_bad += (not ok)
            if U in cyc_targets:
                n_cyc += 1; n_cyc_ok += ok
            if not ok and verbose and n_bad <= 2:
                print("  MISMATCH V=%s U=%s (cyclic: %s)" % (sorted(V), sorted(U), U in cyc_targets))
                for k in sorted(set(p) | set(t), key=str):
                    if p.get(k, 0) != t.get(k, 0):
                        mv = {s + 1: tt for s, tt in enumerate(k[0]) if s + 1 != tt}
                        print("       2*pred %s 2*true %s  %s dots %s sigma %s" % (p.get(k, 0), t.get(k, 0),
                              "*".join(n_ if pw == 1 else "%s^%d" % (n_, pw) for n_, pw in k[1]), k[2], mv))
    return (n_ok, n_bad, n_cyc_ok, n_cyc, failed)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    for p in args:
        res = {v: compare(p, variant=v, verbose=verbose) for v in (("mix",) if verbose else ("mix", "endL", "roleL"))}
        print("%-46s" % os.path.basename(p).replace("__K_.json", ""), "  ".join(
            "%s: wrong %d, cyclic ok %d/%d%s" % (v, r[1], r[2], r[3], " (unrealisable)" if r[4] else "") for v, r in res.items()))
