"""
01_minuscule_scan.py   (sage -python proofs/checks/01_minuscule_scan.py TYPE RANK K [MAXSLOTS])

For EVERY weight mu of the minuscule representation V(omega_K) (= every order ideal of the
minuscule heap) with at most MAXSLOTS slots: one word for the ideal, whether the stack is
cyclic (a cyclic configuration exists), and whether some admissible configuration has a BAD
triple (01_bad_search.bad_triples) -- and if so whether all bad triples are shielded.
Both properties depend only on the ideal (heap invariance, Proposition 7 and Remark B.5),
and both are inherited by larger ideals, so ideals containing a cyclic one are skipped
(marked "cyclic (inherited)").
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from sage.all import RootSystem, CartanType   # noqa: E402
from formula import Stack                      # noqa: E402
from cyclefree import find_cycle               # noqa: E402

_spec = importlib.util.spec_from_file_location("bs01", os.path.join(HERE, "01_bad_search.py"))
bs01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs01)


def ideals(ct, k, maxslots):
    """dict weight -> (word, root-coefficient vector of lambda - mu), BFS from the highest weight."""
    Phi = RootSystem(CartanType(ct))
    L = Phi.weight_lattice()
    alpha = L.simple_roots()
    ac = L.simple_coroots()
    I = list(Phi.cartan_type().index_set())
    lam = L.fundamental_weight(k)
    seen = {lam: ((), tuple(0 for _ in I))}
    frontier = [lam]
    while frontier:
        new = []
        for mu in frontier:
            word, vec = seen[mu]
            if len(word) >= maxslots:
                continue
            for i in I:
                if mu.scalar(ac[i]) == 1:
                    nu = mu - alpha[i]
                    if nu not in seen:
                        v = list(vec)
                        v[I.index(i)] += 1
                        seen[nu] = (word + (i,), tuple(v))
                        new.append(nu)
        frontier = new
    return seen


def main():
    ct = [sys.argv[1], int(sys.argv[2])]
    k = int(sys.argv[3])
    maxslots = int(sys.argv[4]) if len(sys.argv) > 4 else 99
    ids = ideals(ct, k, maxslots)
    order = sorted(ids.values(), key=lambda wv: len(wv[0]))
    cyclic_vecs = []
    stats = {"cycle-free, no bad": 0, "cycle-free, bad (all shielded)": 0, "cycle-free, bad UNSHIELDED": 0,
             "cyclic": 0, "cyclic (inherited)": 0}
    t01 = bs01._t01()
    for word, vec in order:
        if not word:
            continue
        if any(all(a <= b for a, b in zip(cv, vec)) for cv in cyclic_vecs):
            stats["cyclic (inherited)"] += 1
            continue
        st = Stack(ct, word)
        cyc = find_cycle(st, max_pairs=len(word) // 2)
        if cyc is not None:
            cyclic_vecs.append(vec)
            stats["cyclic"] += 1
            continue
        bad = []
        for j, F in bs01.configurations(st):
            if bs01.bad_triples(st, j, F):
                bad.append((j, F))
        if not bad:
            stats["cycle-free, no bad"] += 1
            continue
        unsh = False
        for j, F in bad:
            I = {j} | {b for a, b in F}
            for typ, names, sh in bs01.shields(st, I, j, F, t01):
                if not sh:
                    unsh = True
        key = "cycle-free, bad UNSHIELDED" if unsh else "cycle-free, bad (all shielded)"
        stats[key] += 1
        print("   cycle-free ideal with bad triples:", "".join(map(str, word)), "configurations with bad triples:", len(bad), key, flush=True)
    print("%s%d omega_%d, ideals with <= %d slots: %s" % (ct[0], ct[1], k, maxslots, stats), flush=True)


if __name__ == "__main__":
    main()
