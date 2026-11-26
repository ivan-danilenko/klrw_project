"""
Exterior-algebra model of the omega_1 I-brane.  Each colour c has two odd generators a_c, g_c;
state B_R <-> 1,  A <-> a_c,  C <-> g_c,  B_L <-> a_c g_c.   A vertex is the wedge monomial of
its colours.  Terms are built from contractions i(a_c), i(g_c), wedging e(g_c) and the
replacement rho_c = "a_c -> g_c".   We compare the resulting Koszul signs with the data.
"""
import sys, os, itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega1_table import load, fmt
from omega1_rule import targets, vertices

def monomial(n, v):
    k, A, C = v
    gens = set()
    for c in range(1, n + 1):
        if c in A: gens.add(("a", c))
        elif c in C: gens.add(("g", c))
        elif c <= k: gens.add(("a", c)); gens.add(("g", c))
    return gens

def contract(gens, order, x):
    """Remove generator x from the sorted wedge monomial: sign = (-1)^(number of generators before x)."""
    pos = sum(1 for y in gens if order(y) < order(x))
    return gens - {x}, (-1) ** pos

def wedge(gens, order, x):
    pos = sum(1 for y in gens if order(y) < order(x))
    return gens | {x}, (-1) ** pos

def replace(gens, order, c):
    g1, s1 = contract(gens, order, ("a", c))
    g2, s2 = wedge(g1, order, ("g", c))
    return g2, s1 * s2

def koszul_sign(n, v, tag, order):
    k, A, C = v
    gens = monomial(n, v); s = 1
    if tag[0] == "int":
        c = tag[1]
        gens, s = contract(gens, order, ("g", c))
        return s
    d = tag[1]; l = k - d; sw = list(range(l + 1, k + 1))
    ops = []
    if tag[0] == "i":
        ops = [("rho", c) for c in sw[:-1]] + [("ia", k)]
    elif tag[0] == "ii":
        ops = [("ia", l + 1)] + [("rho", c) for c in sw[1:]]
    else:
        c = tag[2]
        if d == 1:
            ops = [("ig", k), ("ia", k), ("eg", c)]
        else:
            ops = [("ia", l + 1)] + [("rho", x) for x in sw[1:-1]] + [("ia", k), ("eg", c)]
    for op, c in ops:
        if op == "rho": gens, t = replace(gens, order, c)
        elif op == "ia": gens, t = contract(gens, order, ("a", c))
        elif op == "ig": gens, t = contract(gens, order, ("g", c))
        else: gens, t = wedge(gens, order, ("g", c))
        s *= t
    return s

ORDERS = {
    "a1 g1 a2 g2 ..": lambda n: (lambda y: 2 * y[1] + (0 if y[0] == "a" else 1)),
    "g1 a1 g2 a2 ..": lambda n: (lambda y: 2 * y[1] + (1 if y[0] == "a" else 0)),
    "a1..an g1..gn": lambda n: (lambda y: y[1] + (0 if y[0] == "a" else n + 1)),
    "a1..an gn..g1": lambda n: (lambda y: y[1] if y[0] == "a" else 2 * n + 2 - y[1]),
    "an..a1 g1..gn": lambda n: (lambda y: -y[1] if y[0] == "a" else y[1] + 1),
    "g1..gn a1..an": lambda n: (lambda y: y[1] + (0 if y[0] == "g" else n + 1)),
}

if __name__ == "__main__":
    for oname, mk in ORDERS.items():
        ratios = defaultdict(set)
        for n in (2, 3, 4, 5):
            d, P, V, terms = load(n)
            sign = {(t[0], t[1]): t[3] for t in terms}
            order = mk(n)
            for v in vertices(n):
                for w, tag in targets(n, v):
                    ks = koszul_sign(n, v, tag, order)
                    typ = tag[0] if tag[0] != "int" else ("intBA" if tag[1] <= v[0] else "intCB")
                    if tag[0] == "iii":
                        typ = "iiiL" if tag[2] <= v[0] - tag[1] else "iiiR"
                    dd = 0 if tag[0] == "int" else tag[1]
                    ratios[(typ, n, v[0], dd)].add(sign[(v, w)] * ks)
        bad = sum(1 for r in ratios.values() if len(r) > 1)
        print("order %-18s classes (type, n, k, d) with non-constant ratio: %d of %d" % (oname, bad, len(ratios)))
        if bad == 0:
            for key in sorted(ratios):
                print("     ", key, ratios[key])
