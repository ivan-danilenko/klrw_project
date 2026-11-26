"""
Does a stack admit configurations (mover, F) whose link graph is connected, has no
blocked pairs, and contains a cycle?  (For those the tree formula is incomplete.)
Pure Python:  python3 cyclefree.py   (uses sage only to produce the words; words are cached in words.json)
"""
import itertools, json, os, sys
from formula import Stack

def links(st, j, F):
    """Return (n_links, connected, blocked) of the link graph of (j, F)."""
    cj = st.col[j - 1]
    adj = {x: set() for x in ["m"] + list(F)}
    n = 0
    for p in F:
        a, b = p
        if a < j < b:
            if st.col[a - 1] == cj:
                return None
            if st.adjacent(a, j):
                adj["m"].add(p); adj[p].add("m"); n += 1
    for p, q in itertools.combinations(F, 2):
        (a, b), (a2, b2) = sorted((p, q))
        if a < a2 < b < b2:
            if st.col[a - 1] == st.col[a2 - 1]:
                return None
            if st.adjacent(a, a2):
                adj[p].add(q); adj[q].add(p); n += 1
    seen = {"m"}; stack = ["m"]
    while stack:
        for y in adj[stack.pop()]:
            if y not in seen:
                seen.add(y); stack.append(y)
    return n, len(seen) == len(F) + 1

def find_cycle(st, max_pairs=6):
    """Smallest connected unblocked configuration with more links than pairs, or None."""
    best = None
    for j in range(1, st.m + 1):
        avail = [p for p in st.pairs if j not in p]
        # grow connected configurations
        frontier = {()}
        seen = {()}
        for size in range(1, max_pairs + 1):
            new = set()
            for F in frontier:
                used = {x for p in F for x in p}
                for p in avail:
                    if p in F or p[0] in used or p[1] in used:
                        continue
                    G = tuple(sorted(F + (p,)))
                    if G in seen:
                        continue
                    seen.add(G)
                    res = links(st, j, G)
                    if res is None or not res[1]:
                        continue
                    if res[0] > len(G):
                        if best is None or len(G) < len(best[1]):
                            best = (j, G, res[0])
                        continue          # do not grow beyond a cycle
                    new.add(G)
            frontier = new
            if best is not None and len(best[1]) <= size:
                break
    return best

if __name__ == "__main__":
    words = json.load(open("words.json"))
    for name, (ct, red, word) in sorted(words.items()):
        st = Stack(ct, word)
        mult = max(len(v) for v in st.same.values())
        best = find_cycle(st)
        if best is None:
            print("%-10s %2d strands, max colour multiplicity %d: no cyclic configuration -> tree formula complete" % (name, len(word), mult))
        else:
            j, F, n = best
            need = max(max(p) for p in F + ((j, j),))
            print("%-10s %2d strands, max colour multiplicity %d: CYCLE, e.g. mover %d (c%d), pairs %s; first %d slots suffice" % (
                name, len(word), mult, j, st.col[j - 1], ["(%d,%d)c%d" % (a, b, st.col[a - 1]) for a, b in F], need))
