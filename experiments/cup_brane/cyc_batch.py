"""
Enumerate cyclic configurations of a stack, pick minimal source vertices, and list them
with the size of their down-sets.   python3 cyc_batch.py NAME [max_pairs] [max_vertices]
(NAME as in words.json, e.g. D6_w6).  Prints interval.py command lines.
"""
import sys, json, itertools
from formula import Stack
from interval import dominated_sets  # noqa  (pure python part)

def all_configs(st, max_pairs):
    from cycle import link_list
    res = []
    for j in range(1, st.m + 1):
        avail = [p for p in st.pairs if j not in p]
        found = {(): []}
        frontier = [()]
        for size in range(max_pairs):
            new = []
            for F in frontier:
                used = {x for p in F for x in p}
                for p in avail:
                    if p in F or p[0] in used or p[1] in used:
                        continue
                    G = tuple(sorted(F + (p,)))
                    if G in found:
                        continue
                    links = link_list(st, j, G)
                    if links is None:
                        continue
                    adj = {x: set() for x in ("m",) + G}
                    for x, y, _, _ in links:
                        adj[x].add(y); adj[y].add(x)
                    seen = {"m"}; todo = ["m"]
                    while todo:
                        for y in adj[todo.pop()]:
                            if y not in seen:
                                seen.add(y); todo.append(y)
                    if len(seen) != len(G) + 1:
                        continue
                    found[G] = links
                    new.append(G)
            frontier = new
        for F, links in found.items():
            b1 = len(links) - len(F)
            if b1 >= 1:
                res.append((j, F, links, b1))
    return res

def downset_size(st, V):
    n = 1
    for (p, c), slots in st.same.items():
        n *= len(dominated_sets(slots, [s for s in slots if s in V], []))
    return n

if __name__ == "__main__":
    name = sys.argv[1]
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    max_vertices = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    ct, red, word = json.load(open("words.json"))[name]
    st = Stack(ct, word)
    seen = {}
    for j, F, links, b1 in all_configs(st, max_pairs):
        V = frozenset({j} | {b for a, b in F})
        # minimal cyclic core only: skip configurations with a leaf pair (they contain a smaller cyclic one)
        deg = {x: 0 for x in ("m",) + F}
        for x, y, _, _ in links:
            deg[x] += 1; deg[y] += 1
        if any(deg[p] == 1 for p in F):
            continue
        on_cycle_m = deg["m"] >= 2
        size = downset_size(st, V)
        key = V
        if key not in seen or seen[key][0] > size:
            seen[key] = (size, j, F, b1, on_cycle_m, len(links))
    rows = sorted(seen.items(), key=lambda kv: kv[1][0])
    print("# %s word %s: %d minimal cyclic source vertices" % (name, "".join(map(str, word)), len(rows)))
    for V, (size, j, F, b1, on_m, nl) in rows:
        flag = "" if size <= max_vertices else "   # too big"
        print("%s %d %s full \"%s\" \"[]\"   # %d vertices, mover %d, pairs %s, b1=%d, cycle through mover: %s%s" % (
            ct[0], ct[1], "".join(map(str, word)), sorted(V), size, j, list(F), b1, on_m, flag))
