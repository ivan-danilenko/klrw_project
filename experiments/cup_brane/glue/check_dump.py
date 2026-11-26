"""
Independent check of a dump written by glue.py: rebuild the glued complex
from the JSON alone (over the Laurent ring with base QQ), check that every
entry is homogeneous of the right degree, and construct a KLRWPerfectComplex
(its constructor asserts d^2 = 0).  Also checks that the off-diagonal blocks
of the final differential coincide with the recorded gluing maps.

    sage -python glue/check_dump.py glue/data/A2_w1.json ...
"""
import sys
import os
import json
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patches  # noqa: E402

patches.install()

from sage.combinat.root_system.cartan_type import CartanType  # noqa: E402
from sage.rings.integer_ring import ZZ  # noqa: E402
from sage.rings.rational_field import QQ  # noqa: E402
from sage.matrix.constructor import matrix  # noqa: E402

from klrw.framed_dynkin import FramedDynkinDiagram_with_dimensions, NodeInFramedQuiver  # noqa: E402
from klrw.klrw_algebra import KLRWAlgebra  # noqa: E402
from klrw.perfect_complex import KLRWPerfectComplex, KLRWIrreducibleProjectiveModule  # noqa: E402


def node_from_name(name):
    framing = name[0] == "W"
    return NodeInFramedQuiver(int(name[1:]), framing=framing)


def algebra_for(D, invertible=True, base=QQ):
    ct = CartanType(D["cartan_type"])
    DD = FramedDynkinDiagram_with_dimensions(ct)
    counts = defaultdict(int)
    for name in D["projectives"][0]["state"]:
        counts[name] += 1
    for name, c in counts.items():
        DD[node_from_name(name)] += c
    for v in DD.quiver.vertices():
        DD.dimensions_dict[v] += 0
    return KLRWAlgebra(base, DD, dot_scaling=True, edge_scaling=True, vertex_scaling=True,
                       invertible_parameters=invertible, warnings=True)


def check(path, polynomial_model=False):
    """
    polynomial_model=True: first rescale every envelope by the unit monomial
    found by analyze.nonneg_rescaling (all exponents become >= 0), then rebuild
    the complex over the KLRW algebra WITHOUT inverted parameters, base ZZ.
    """
    t0 = time.time()
    D = json.load(open(path))
    if polynomial_model:
        import glue_analyze as analyze
        glue_terms = [t for t in D["terms"] if t["src_envelope"] != t["tgt_envelope"]]
        res = analyze.nonneg_rescaling(D, glue_terms, D["variables"])
        assert res is not None, "no non-negative rescaling"
        resc, new_glue = res
        D["terms"] = [t for t in D["terms"] if t["src_envelope"] == t["tgt_envelope"]] + new_glue
        assert all(e >= 0 for t in D["terms"] for e in t["params"].values())
        KLRW = algebra_for(D, invertible=False, base=ZZ)
        Gr = KLRW.grading_group
        Dalg = KLRW.base()
        var_deg = {x: Dalg.element_degree(Dalg(x), Gr) for x in D["variables"]}
    else:
        KLRW = algebra_for(D)
    assert list(KLRW.base().variable_names()) == D["variables"]
    dots_alg = KLRW.base()
    G = KLRW.grading_group
    label_of = {repr(l): l for l in G.basis().keys()}
    State = KLRW.KLRWBraid.KLRWstate_set
    Braid = KLRW.KLRWBraid

    def state(names):
        return State._element_constructor_(tuple(node_from_name(n) for n in names))

    def degree(dd):
        return G.sum_of_terms((label_of[k], v) for k, v in dd.items())

    def env_shift(k):
        if not polynomial_model:
            return G.zero()
        return G.sum([e * var_deg[x] for x, e in resc[k].items()])

    P = D["projectives"]
    by_h = defaultdict(list)
    for p in P:
        by_h[p["hom_deg"]].append(p)
    projectives = {}
    for h, lst in by_h.items():
        lst.sort(key=lambda p: p["position"])
        projectives[h] = [KLRWIrreducibleProjectiveModule(
            state(p["state"]), degree(p["equivariant_degree_dict"]) + env_shift(p["envelope"])) for p in lst]

    def term_element(t):
        right = state(t["right_state"])
        braid = Braid._element_constructor_(right, tuple(t["word"]))
        assert [repr(x) for x in braid.left_state()] == t["left_state"]
        coeff = dots_alg(ZZ(t["coeff"]) if polynomial_model else QQ(t["coeff"]))
        for name, e in list(t["params"].items()) + list(t["dots"].items()):
            coeff *= dots_alg(name) ** e
        return KLRW.term(braid, coeff)

    dd = D["differential_degree"]
    entries = defaultdict(lambda: defaultdict(lambda: KLRW.zero()))
    for t in D["terms"]:
        s, g = P[t["source"]], P[t["target"]]
        assert g["hom_deg"] == s["hom_deg"] + dd
        entries[s["hom_deg"]][g["position"], s["position"]] += term_element(t)
    n_hom = 0
    for h, e in entries.items():
        for (row, col), el in e.items():
            deg = el.degree(check_if_homogeneous=True)
            assert deg == projectives[h + dd][row].equivariant_degree - projectives[h][col].equivariant_degree
            n_hom += 1
    diff = {
        h: matrix(KLRW.opposite, len(projectives[h + dd]), len(projectives[h]), dict(e), sparse=True)
        for h, e in entries.items()
    }
    KLRWPerfectComplex(ring=KLRW, projectives=projectives, differential=diff, differential_degree=dd)
    if polynomial_model:
        print("{}: rescaled model over the polynomial (non-inverted) algebra, base ZZ: {} projectives, "
              "{} entries homogeneous, d^2 = 0; envelope rescalings {}  [{:.1f}s]".format(
                  os.path.basename(path), len(P), n_hom,
                  ["*".join("{}^{}".format(x, e) for x, e in sorted(r.items())) or "1" for r in resc],
                  time.time() - t0), flush=True)
        return

    # off-diagonal part of the final complex versus the recorded gluing maps
    def key(t, src_eps, tgt_eps):
        return (t["src_envelope"], tuple(src_eps), t["tgt_envelope"], tuple(tgt_eps), tuple(t["word"]),
                tuple(sorted(t["params"].items())), tuple(sorted(t["dots"].items())))

    final = {}
    for t in D["terms"]:
        if t["src_envelope"] != t["tgt_envelope"]:
            final[key(t, P[t["source"]]["eps"], P[t["target"]]["eps"])] = QQ(t["coeff"])
    maps = {}
    for st in D["steps"]:
        for t in st.get("map_terms", []):
            maps[key(t, t["src_eps"], t["tgt_eps"])] = QQ(t["coeff"])
    ratios = defaultdict(int)
    for k in set(final) | set(maps):
        if k in final and k in maps:
            ratios[str(final[k] / maps[k])] += 1
        else:
            ratios["only in " + ("final" if k in final else "maps")] += 1
    print("{}: {} projectives, {} entries homogeneous, d^2 = 0 (rebuilt from JSON); "
          "final off-diagonal / recorded map coefficient ratios: {}  [{:.1f}s]".format(
              os.path.basename(path), len(P), n_hom, dict(ratios), time.time() - t0), flush=True)


if __name__ == "__main__":
    paths = [p for p in sys.argv[1:] if not p.startswith("--")]
    for path in paths:
        check(path, polynomial_model="--polynomial" in sys.argv)
