"""
Build the glued complex (I-brane) for sl_{n+1}, omega_1 from the closed rule and check it.

    sage -python glue/omega1_build.py N

Vertices (k; A, C); terms from omega1_rule.targets; signs from the exterior-algebra model
(generators a_1..a_n, g_n..g_1; state B_R = 1, A = a_c, C = g_c, B_L = a_c g_c):
    internal        i(g_c)
    (i)             (-1)^(d+1) * i(a_k) prod_{l<c<k} rho_c
    (ii)            prod_{l+1<c<=k} rho_c  i(a_{l+1})
    (iii)           (-1)^(delta [+ d if the extra colour is on the left]) * e(g_c) i(a_k) [prod rho] i(a_{l+1})
                    (d = 1:  e(g_c) i(g_k) i(a_k)),   delta = distance of the extra colour from the block
Coefficients: the unique Laurent monomial in t, u of the right multidegree.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import patches
patches.install()
from cupbrane import *
from cupbrane import _to_csc
from omega1_rule import targets, vertices
from omega1_koszul import monomial, contract, wedge, replace
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.rings.rational_field import QQ


def order_fn(n):
    return lambda y: y[1] if y[0] == "a" else 2 * n + 2 - y[1]


def rule_sign(n, v, tag):
    k, A, C = v
    order = order_fn(n)
    gens = monomial(n, v)
    if tag[0] == "int":
        return contract(gens, order, ("g", tag[1]))[1]
    d = tag[1]; l = k - d; sw = list(range(l + 1, k + 1))
    if tag[0] == "i":
        ops = [("rho", c) for c in sw[:-1]] + [("ia", k)]
        s = (-1) ** (d + 1)
    elif tag[0] == "ii":
        ops = [("ia", l + 1)] + [("rho", c) for c in sw[1:]]
        s = 1
    else:
        c = tag[2]
        if d == 1:
            ops = [("ia", k), ("ig", k), ("eg", c)]
        else:
            ops = [("ia", l + 1)] + [("rho", x) for x in sw[1:-1]] + [("ia", k), ("eg", c)]
        if c <= l:
            s = (-1) ** ((l + 1 - c) + d)
        else:
            s = (-1) ** (c - k)
    for op, c in ops:
        if op == "rho": gens, t = replace(gens, order, c)
        elif op == "ia": gens, t = contract(gens, order, ("a", c))
        elif op == "ig": gens, t = contract(gens, order, ("g", c))
        else: gens, t = wedge(gens, order, ("g", c))
        s *= t
    return s


def build(n, verbose=True):
    t0 = time.time()
    ct = CartanType(["A", n])
    DD = FramedDynkinDiagram_with_dimensions(ct)
    W1 = NodeInFramedQuiver(1, framing=True); Wn = NodeInFramedQuiver(n, framing=True)
    for v in DD.quiver.vertices():
        DD[v] = 0
    DD[W1] += 1; DD[Wn] += 1
    for c in range(1, n + 1):
        DD[NodeInFramedQuiver(c)] += 1
    KLRW = KLRWAlgebra(ZZ, DD, dot_scaling=True, edge_scaling=True, vertex_scaling=True,
                       invertible_parameters=True, warnings=True)
    D = KLRW.base()
    G = KLRW.grading_group
    Braid = KLRW.KLRWBraid
    State = Braid.KLRWstate_set

    def marked(v):
        k, A, C = v
        B = [c for c in range(1, n + 1) if c not in A and c not in C]
        return tuple(sorted(A, reverse=True)) + ("Wa",) + tuple(B) + ("Wb",) + tuple(sorted(C, reverse=True))

    def state(v):
        return State._element_constructor_(tuple(
            W1 if x == "Wa" else Wn if x == "Wb" else NodeInFramedQuiver(x) for x in marked(v)))

    def braid_elem(v, w):
        word = braid_from_marked_state(marked(v), marked(w))
        return KLRW.monomial(Braid._element_constructor_(state(w), word))

    verts = list(vertices(n))
    terms = [(v, w, tag) for v in verts for w, tag in targets(n, v)]
    # equivariant degrees: internal terms and the d = 1 terms of type (i) have coefficient 1
    deg = {}
    start = (0, frozenset(), frozenset())
    deg[start] = G.zero()
    unit = [(v, w) for v, w, tag in terms if tag[0] == "int" or (tag[0] == "i" and tag[1] == 1)]
    bd = {}
    changed = True
    while changed:
        changed = False
        for v, w in unit:
            if (v, w) not in bd:
                bd[(v, w)] = braid_elem(v, w).degree(check_if_homogeneous=True)
            if w in deg and v not in deg:
                deg[v] = deg[w] - bd[(v, w)]; changed = True
            elif v in deg and w not in deg:
                deg[w] = deg[v] + bd[(v, w)]; changed = True
    assert len(deg) == len(verts), "degrees do not propagate to all vertices"
    for v, w in unit:
        assert deg[w] - deg[v] == bd[(v, w)], "inconsistent degrees"
    # Laurent monomials in the parameters by degree
    names = [nm for nm in D.variable_names() if not nm.startswith("x_")]
    gens = [D(nm) for nm in names]
    M = matrix(QQ, [list(D.element_degree(g, G)._vector_()) for g in gens]).transpose()
    def coefficient(delta):
        if delta == G.zero():
            return D.one()
        x = M.solve_right(vector(QQ, list(delta._vector_())))
        assert all(c in ZZ for c in x), "non-integral exponent"
        assert M.right_kernel().dimension() == 0
        out = D.one()
        for g, e in zip(gens, x):
            if e:
                out *= g ** int(e)
        return out
    H = lambda v: v[0] + sum(1 for c in range(1, v[0] + 1) if c not in v[1]) + len(v[2])
    by_h = defaultdict(list)
    for v in verts:
        by_h[H(v)].append(v)
    index = {v: (h, i) for h, lst in by_h.items() for i, v in enumerate(lst)}
    projectives = {h: [KLRWIrreducibleProjectiveModule(state(v), deg[v]) for v in lst] for h, lst in by_h.items()}
    diff = {h: {} for h in by_h if h - 1 in by_h}
    monos = defaultdict(int)
    for v, w, tag in terms:
        h, col = index[v]; h2, row = index[w]
        assert h2 == h - 1, (v, w, tag)
        b = braid_elem(v, w)
        try:
            coeff = coefficient(deg[w] - deg[v] - b.degree(check_if_homogeneous=True))
        except Exception as err:
            raise RuntimeError("no coefficient for %s -> %s %s: %s" % (v, w, tag, err))
        monos[str(coeff)] += 1
        entry = (rule_sign(n, v, tag) * coeff) * b
        assert (row, col) not in diff[h]
        diff[h][row, col] = entry
    d_csc = _to_csc(KLRW, projectives, diff)
    ok = check_square_zero(KLRW, projectives, d_csc)
    if verbose:
        print("sl_%d omega_1: %d projectives, %d terms, %d distinct coefficients, d^2 = 0: %s  [%.0fs]" % (
            n + 1, len(verts), len(terms), len(monos), ok, time.time() - t0), flush=True)
    return ok, KLRW, projectives, diff, monos


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]]:
        build(n)
