"""
Which weights mu of a minuscule representation V_lambda have a stack (chain lambda -> mu)
that admits a cyclic configuration?  The property depends only on the order ideal of the
minuscule poset (i.e. on mu) and is inherited by larger ideals, so we list the minimal ones.

    sage -python cyclic_weights.py TYPE RANK NODE [max_pairs]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sage.all
from sage.combinat.root_system.root_system import RootSystem
from sage.combinat.root_system.cartan_type import CartanType
from formula import Stack
from cyclefree import find_cycle


def main(ct, k, max_pairs=4):
    Phi = RootSystem(CartanType(ct))
    L = Phi.weight_lattice()
    alpha = L.simple_roots(); acheck = L.simple_coroots()
    index = Phi.cartan_type().index_set()
    lam = L.fundamental_weight(k)
    # breadth-first over the orbit, remembering one word for each weight
    words = {lam: ()}
    frontier = [lam]
    order = [lam]
    while frontier:
        new = []
        for mu in frontier:
            for i in index:
                if mu.scalar(acheck[i]) == 1:
                    nu = mu - alpha[i]
                    if nu not in words:
                        words[nu] = words[mu] + (i,)
                        new.append(nu); order.append(nu)
        frontier = new
    cyclic = {}
    for mu in order:
        w = words[mu]
        if len(w) < 7:
            cyclic[mu] = None
            continue
        # inherited from a smaller ideal?
        inherited = False
        for i in index:
            up = mu + alpha[i]
            if up in words and cyclic.get(up):
                inherited = True
                break
        if inherited:
            cyclic[mu] = "inherited"
            continue
        best = find_cycle(Stack(ct, w), max_pairs)
        cyclic[mu] = (best, w) if best else None
    minimal = [(mu, v) for mu, v in cyclic.items() if v and v != "inherited"]
    n_cyc = sum(1 for v in cyclic.values() if v)
    print("%s%d, omega_%d: %d weights, %d with a cyclic stack, %d minimal ones" % (ct[0], ct[1], k, len(order), n_cyc, len(minimal)))
    for mu, ((j, F, nl), w) in minimal:
        diff = lam - mu
        coeffs = {}
        for c in w:
            coeffs[c] = coeffs.get(c, 0) + 1
        print("   mu = %s   lambda - mu = %s   (%d strands)   e.g. mover %d (c%d), pairs %s" % (
            mu, " + ".join("%d a%d" % (coeffs[i], i) for i in sorted(coeffs)), len(w), j, w[j - 1],
            ["(%d,%d)c%d" % (a, b, w[a - 1]) for a, b in F]))
    return cyclic, words


if __name__ == "__main__":
    ct = [sys.argv[1], int(sys.argv[2])]
    main(ct, int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 4)
