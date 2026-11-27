"""
08 -- analysis of the cyclic terms of a stored differential (proofs/08_cyclic_normal_form.md, §5).

    python3 08_analyse.py 08_data/FILE.json [...]

For every cyclic configuration (j, F) whose link graph is a single 4-cycle (3 pairs) and every vertex
I at which it is admissible (j, b_p in I, a_p not in I), lists the spanning trees T_e = C - e with their
tree sign (formula.Stack.sign) and monomial, and the dotted terms of the stored differential in the
component I -> J(j, F) with the bijection sigma(j, F), with the dots translated to slots.
Pure Python.
"""
import os
import sys
import json
import itertools
import importlib
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
C8 = importlib.import_module("08_comb")


def load(path):
    data = json.load(open(path))
    W = C8.WordData(data["cartan_type"], data["red"], data["word"])
    terms = defaultdict(list)
    for src, tgt, sg, mono, c in data["terms"]:
        terms[frozenset(src), frozenset(tgt)].append((tuple(sg), mono, c))
    return W, data, terms


def dot_slot(W, I, name):
    """klrw dot name x_c_k -> slot (k-th strand of colour c from the left at the source e_I)."""
    _, c, k = name.split("_")
    c, k = int(c), int(k)
    labs = [l for l in W.labels(I) if l != "W" and W.c(l) == c]
    return labs[k - 1]


def links_of(W, j, F):
    """links of the configuration: list of (u, v, connector-free description); nodes 'M' or pairs."""
    out = []
    for p in F:
        a, b = p
        if a < j < b and W.adj(W.c(a), W.c(j)):
            out.append(("M", p))
    for p, q in itertools.combinations(F, 2):
        (a, b), (a2, b2) = sorted((p, q))
        if a < a2 < b < b2 and W.adj(W.c(a), W.c(a2)):
            out.append((tuple(sorted((p, q)))))
    return [tuple(e) for e in out]


def tree_parent(W, j, F, links):
    """parent map of the tree (j, F) with the given links (rooted at the mover), with connectors."""
    adj = defaultdict(list)
    for u, v in links:
        adj[u].append(v)
        adj[v].append(u)
    parent = {}
    seen = {"M"}
    stack = ["M"]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            a, b = v
            if u == "M":
                conn = j
            else:
                a2, b2 = u
                conn = a2 if a < a2 < b else b2
                assert a < conn < b
            parent[v] = (u if u != "M" else "mover", conn)
            stack.append(v)
    return parent


def spanning_data(W, I, j, F):
    links = links_of(W, j, F)
    out = []
    for e in links:
        rest = [x for x in links if x != e]
        par = tree_parent(W, j, F, rest)
        if len(par) != len(F):
            continue
        sign = W.stack.sign(I, j, F, par)
        mono = W.stack.monomial(par)
        out.append((e, sign, mono, par))
    return links, out


def node_name(W, v, j):
    if v == "M":
        return "M%d(c%d)" % (j, W.c(j))
    return "(%d,%d)c%d" % (v[0], v[1], W.c(v[0]))


def analyse(path, verbose=True):
    W, data, terms = load(path)
    cyc = set()
    # all cyclic configurations with 3 pairs (4-cycles through the mover)
    for j, F in W.minimal_cyclic():
        if len(F) == 3:
            cyc.add((j, F))
    rep = []
    for j, F in sorted(cyc):
        Ibase = {j} | {b for a, b in F}
        others = [s for s in W.slots if s not in Ibase and s not in {a for a, b in F}]
        for k in range(len(others) + 1):
            for extra in itertools.combinations(others, k):
                I = frozenset(Ibase | set(extra))
                J = frozenset((I - {j} - {b for a, b in F}) | {a for a, b in F})
                sigma = list(W.slots)
                for a, b in F:
                    sigma[a - 1], sigma[b - 1] = b, a
                sigma = tuple(sigma)
                links, sp = spanning_data(W, I, j, F)
                comp = terms.get((I, J), [])
                dotted = [(m, c) for (sg, m, c) in comp if sg == sigma and any(n.startswith("x_") for n in m)]
                plain = [(sg, m, c) for (sg, m, c) in comp if sg != sigma]
                if verbose:
                    print("== %s red %d word %s: mover %d, pairs %s, I = %s -> J = %s" % (
                        W.ct, W.red, "".join(map(str, W.col)), j, F, sorted(I), sorted(J)))
                    for e, sign, mono, par in sp:
                        print("     T - %-32s sign %+d  %s" % (" -- ".join(node_name(W, v, j) for v in e), sign, C8.mono_str(mono)))
                    for m, c in sorted(dotted, key=lambda x: sorted(x[0].items())):
                        params = tuple(sorted((n, p) for n, p in m.items() if not n.startswith("x_")))
                        dots = ["x[%d]" % dot_slot(W, I, n) + ("^%d" % p if p > 1 else "") for n, p in m.items() if n.startswith("x_")]
                        print("     dotted %+d %-36s %s" % (c, C8.mono_str(params), "*".join(dots)))
                    for sg, m, c in plain:
                        print("     same component, other sigma %+d %-30s %s" % (c, C8.mono_str(tuple(sorted(m.items()))), W.cycle_str(sg)))
                rep.append((I, J, sp, dotted, plain))
    return rep


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyse(p)
