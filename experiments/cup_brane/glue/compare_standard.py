"""
Run klrw.standard_ebranes.standard_ebranes itself (only the solver patches of
patches.py installed) and compare its output term by term with a dump of
glue.py.  Only works where standard_ebranes does not hit the quiver-data
coercion bug (A1, A2 omega_1); see glue.py.

    sage -python glue/compare_standard.py A 2 1
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patches  # noqa: E402

patches.install()

from sage.combinat.root_system.root_system import RootSystem  # noqa: E402
from sage.combinat.root_system.cartan_type import CartanType  # noqa: E402
from klrw.standard_ebranes import standard_ebranes  # noqa: E402


def hint(g):
    return int(sum(c for _, c in g))


def main(ctype, rank, node):
    ct = CartanType([ctype, rank])
    Phi = RootSystem(ct)
    E = standard_ebranes(Phi, Phi.weight_lattice().fundamental_weight(node))
    mine = json.load(open(os.path.join(HERE, "data", "{}{}_w{}.json".format(ctype, rank, node))))
    # projectives in the same order?
    theirs_p = []
    for g in sorted(E.projectives().keys(), key=hint):
        for p, pr in enumerate(E.projectives(g)):
            theirs_p.append((hint(g), p, [repr(x) for x in pr.state], str(pr.equivariant_degree)))
    mine_p = [(p["hom_deg"], p["position"], p["state"], p["equivariant_degree"]) for p in mine["projectives"]]
    assert theirs_p == mine_p, "projectives differ"
    names = E.ring().base().variable_names()
    theirs = set()
    for g, mat in E.differential:
        for (i, j), entry in mat.dict(copy=False).items():
            for braid, coeff in entry.value:
                for exp, c in coeff.iterator_exp_coeff():
                    mono = tuple(sorted((names[a], int(e)) for a, e in enumerate(exp) if e))
                    theirs.add((hint(g), int(i), int(j), tuple(int(a) for a in braid.word()), mono, str(c)))
    idx = {p["index"]: p for p in mine["projectives"]}
    ours = set()
    for t in mine["terms"]:
        s, g = idx[t["source"]], idx[t["target"]]
        mono = tuple(sorted(list(t["params"].items()) + list(t["dots"].items())))
        ours.add((s["hom_deg"], g["position"], s["position"], tuple(t["word"]), mono, t["coeff"]))
    print("{}{} w{}: standard_ebranes terms {}, glue.py terms {}, identical: {}".format(
        ctype, rank, node, len(theirs), len(ours), theirs == ours))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
