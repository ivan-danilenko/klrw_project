"""
Candidate extension of the tree formula to configurations whose link graph has cycles.

    sage -python cycle.py data_local/FILE.json

For a connected, unblocked configuration (V; j, F) with link graph G:

    E = kappa * sum_{T spanning tree of G} sign(T) * mono(T) * prod_{e not in T} bigon(e)

where a non-tree link is drawn as a bigon between the two events it joins, averaged over
drawing it on the right or on the left of the red strand.
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cupbrane import *
from fast import braid_for, exp_for
from formula import Stack
from events import diagram, linear_extension
from decode import term_keys
from sage.rings.rational_field import QQ


def link_list(stack, j, F):
    """Links as (x, y, connector_x, connector_y) with x before y on the right-hand side; None if blocked.
    connector_x = slot of y inside the interval of x (if x is a pair), etc."""
    cj = stack.col[j - 1]
    out = []
    for p in F:
        a, b = p
        if a < j < b:
            if stack.col[a - 1] == cj:
                return None
            if stack.adjacent(a, j):
                out.append(("m", p, None, j))          # child p has connector j
    for p, q in itertools.combinations(sorted(F), 2):
        (a, b), (a2, b2) = p, q
        if a < a2 < b < b2:
            if stack.col[a - 1] == stack.col[a2 - 1]:
                return None
            if stack.adjacent(a, a2):
                # a2 lies inside p (connector of p when q is its parent); b lies inside q
                out.append((p, q, a2, b))
    return out


def configurations(stack, V, j, max_b1=1):
    """Connected unblocked configurations F at (V, j) with first Betti number <= max_b1."""
    avail = [(a, b) for a, b in stack.pairs if a not in V and b in V and j not in (a, b)]
    found = {(): []}
    frontier = [()]
    while frontier:
        new = []
        for F in frontier:
            used = {x for p in F for x in p}
            for p in avail:
                if p in F or p[0] in used or p[1] in used:
                    continue
                G = tuple(sorted(F + (p,)))
                if G in found:
                    continue
                links = link_list(stack, j, G)
                if links is None:
                    continue
                # connected?
                adj = {x: set() for x in ("m",) + G}
                for x, y, _, _ in links:
                    adj[x].add(y); adj[y].add(x)
                seen = {"m"}; todo = ["m"]
                while todo:
                    for y in adj[todo.pop()]:
                        if y not in seen:
                            seen.add(y); todo.append(y)
                if len(seen) != len(G) + 1 or len(links) - len(G) > max_b1:
                    continue
                found[G] = links
                new.append(G)
        frontier = new
    return found


def spanning_trees(F, links):
    nodes = ("m",) + tuple(F)
    for subset in itertools.combinations(range(len(links)), len(F)):
        adj = {x: [] for x in nodes}
        for i in subset:
            x, y, cx, cy = links[i]
            adj[x].append((y, cy))     # y child of x: connector of y is cy
            adj[y].append((x, cx))     # x child of y: connector of x is cx
        parent = {}
        seen = {"m"}; todo = ["m"]
        while todo:
            p = todo.pop()
            for child, connector in adj[p]:
                if child not in seen:
                    seen.add(child); parent[child] = (p, connector); todo.append(child)
        if len(seen) == len(nodes):
            yield subset, parent


def cycle_order(F, links):
    """Indices of the links on the unique cycle, in cyclic order, starting with the link at the
    mover (if the mover is on the cycle) that goes to the pair with the smallest a."""
    nodes = ("m",) + tuple(F)
    deg = {x: 0 for x in nodes}
    alive = set(range(len(links)))
    changed = True
    while changed:                       # strip leaves
        changed = False
        deg = {x: 0 for x in nodes}
        for i in alive:
            deg[links[i][0]] += 1; deg[links[i][1]] += 1
        for i in list(alive):
            if deg[links[i][0]] == 1 or deg[links[i][1]] == 1:
                alive.discard(i); changed = True
    cyc_links = sorted(alive)
    # walk around
    start_node = "m" if any("m" in links[i][:2] for i in cyc_links) else min(
        x for i in cyc_links for x in links[i][:2])
    at = [i for i in cyc_links if start_node in links[i][:2]]
    at.sort(key=lambda i: [x for x in links[i][:2] if x != start_node][0])
    order = [at[0]]
    node = [x for x in links[at[0]][:2] if x != start_node][0]
    while len(order) < len(cyc_links):
        nxt = [i for i in cyc_links if node in links[i][:2] and i not in order][0]
        order.append(nxt)
        node = [x for x in links[nxt][:2] if x != node][0]
    return order


def predicted_elements(cube, stack, V, j, cache, max_b1=1, variant="half"):
    """dict U -> KLRW element (over QQ-safe integers times 2**b1 handled by caller)."""
    KLRW = cube.KLRW
    D = KLRW.base()
    out = {}
    for F, links in configurations(stack, V, j, max_b1).items():
        b1 = len(links) - len(F)
        U = (V - {j} - {b for a, b in F}) | {a for a, b in F}
        rel = [(x, y) for x, y, _, _ in links]
        total = KLRW.zero()
        for subset, parent in spanning_trees(F, links):
            mono = stack.monomial(parent)
            sign = stack.sign(V, j, F, parent)
            if mono not in cache:
                cache[mono] = D.monomial(*exp_for(KLRW, mono)) if mono else D.one()
            extra = [i for i in range(len(links)) if i not in subset]
            # every non-tree link: bigon on the right or on the left
            if variant == "half" or not extra:
                side_choices = list(itertools.product("RL", repeat=len(extra)))
            else:
                # alternate around the (unique) cycle; variant = "A" or "B"
                assert len(extra) == 1
                cyc = cycle_order(F, links)
                pos = cyc.index(extra[0])
                first = "R" if variant == "A" else "L"
                other = "L" if first == "R" else "R"
                side_choices = [(first if pos % 2 == 0 else other,), (first if pos % 2 == 0 else other,)]
            for sides in side_choices:
                rel_R = list(rel); rel_L = [(y, x) for x, y in rel]
                for i, side in zip(extra, sides):
                    x, y = rel[i]
                    if side == "R":
                        rel_R[i] = (y, x)
                    else:
                        rel_L[i] = (x, y)
                ro = linear_extension(("m",) + F, rel_R)
                lo = linear_extension(("m",) + F, rel_L)
                if ro is None or lo is None:
                    raise RuntimeError("no consistent time ordering for %s %s" % (F, sides))
                el = diagram(cube, V, j, F, ro, lo)
                total += (2 ** (len(extra)) // 2 ** len(extra)) * sign * cache[mono] * el
        out.setdefault(U, []).append((b1, total))
    return out


def compare(path, max_b1=1, verbose=True, variant="half"):
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
    n_ok = n_bad = n_cyc = 0
    for V in vertices:
        pred = {}
        for j in sorted(V):
            for U, lst in predicted_elements(cube, stack, V, j, cache, max_b1, variant).items():
                if U not in vset:
                    continue
                for b1, el in lst:
                    # el carries a factor 2**b1 (sum over L/R instead of average)
                    for k, c in term_keys(cube, V, U, el).items():
                        d = pred.setdefault(U, {})
                        d[k] = d.get(k, 0) + QQ(c) / 2 ** b1
                    n_cyc += bool(b1)
        for U in set(pred) | set(u for (v, u) in truth if v == V):
            p = {k: c for k, c in pred.get(U, {}).items() if c}
            t = truth.get((V, U), {})
            if p == t:
                n_ok += 1
            else:
                n_bad += 1
                if verbose and n_bad <= 3:
                    print("MISMATCH V=%s U=%s" % (sorted(V), sorted(U)))
                    for k in sorted(set(p) | set(t), key=str):
                        if p.get(k, 0) != t.get(k, 0):
                            print("     pred %s  true %s   %s" % (p.get(k, 0), t.get(k, 0), k))
    print("%s [variant %s]: components equal %d, different %d (cyclic configurations used: %d)" % (os.path.basename(path), variant, n_ok, n_bad, n_cyc))
    return n_bad == 0


if __name__ == "__main__":
    for p in sys.argv[1:]:
        for variant in ("half", "A", "B"):
            compare(p, variant=variant, verbose=False)
