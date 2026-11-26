"""
Gluing over the polynomial (non-inverted) KLRW algebra.

    sage -python glue/glue_poly.py CASE [+|-]

CASE as in glue/data (e.g. A3_w2).  The envelope S_k is shifted, relative to the convention
of standard_ebranes, by the degree of the unit monomial m_k that makes the Laurent result
polynomial (glue_analyze.nonneg_rescaling).  We then recompute, over the polynomial algebra
with base QQ, the spaces of chain maps and of null-homotopic maps at every gluing step.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import patches
patches.install()
import glue as G
import glue_analyze as analyze
from glue import *


def run_poly(case, sign=+1, extra_divide=None):
    D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", case + ".json")))
    glue_terms = [t for t in D["terms"] if t["src_envelope"] != t["tgt_envelope"]]
    resc, _ = analyze.nonneg_rescaling(D, glue_terms, D["variables"])
    if extra_divide:
        for k, mono in extra_divide.items():
            for x, e in mono.items():
                resc[k][x] = resc[k].get(x, 0) + e
    cartan = [D["cartan_type"][0], D["cartan_type"][1]] if isinstance(D["cartan_type"], list) else D["cartan_type"]
    node = D["node"]
    ct = CartanType(cartan)
    Phi = RootSystem(ct)
    lambd = Phi.weight_lattice().fundamental_weight(node)
    pos = weight_poset_in_minuscule_rep(Phi, lambd)
    pos_dual = pos.dual()
    sequences_by_weight = {}
    for mu in pos:
        right_weight_sequence = next(pos_dual.maximal_chains_iterator([mu]))
        right_sequence = tuple(moving_strands_from_weights(iter(right_weight_sequence)))
        left_weight_sequence = next(pos.maximal_chains_iterator([mu]))
        left_sequence = tuple(moving_strands_from_weights(reversed(left_weight_sequence)))
        sequences_by_weight[mu] = (left_sequence, right_sequence)
    (lam_index, _), = tuple(lambd)
    left_framing = NodeInFramedQuiver(lam_index, framing=True)
    right_framing = NodeInFramedQuiver(ct.opposition_automorphism()[lam_index], framing=True)
    DD = FramedDynkinDiagram_with_dimensions(ct)
    DD[left_framing] += 1
    DD[right_framing] += 1
    left_seq, right_seq = next(iter(sequences_by_weight.values()))
    for nd in left_seq + right_seq:
        DD[nd] += 1
    for v in DD.quiver.vertices():
        DD.dimensions_dict[v] += 0
    KLRW = KLRWAlgebra(ZZ, DD, dot_scaling=True, edge_scaling=True, vertex_scaling=True,
                       invertible_parameters=False, warnings=True)
    weights = list(pos_dual.linear_extension())
    assert [str(w) for w in weights] == D["weights_in_gluing_order"]
    stab_ZZ = {mu: stable_envelope(KLRW, left_framing, right_framing,
                                   left_sequence=sequences_by_weight[mu][0], right_sequence=sequences_by_weight[mu][1])
               for mu in weights}
    KLRW_QQ = KLRW._replace_(base_R=QQ)
    stab = {mu: st.base_change(KLRW_QQ) for mu, st in stab_ZZ.items()}
    Gr = KLRW_QQ.grading_group
    Dalg = KLRW_QQ.base()
    def mono_degree(mono):
        return Gr.sum([e * Dalg.element_degree(Dalg(x), Gr) for x, e in mono.items() if e])
    e_brane = stab[weights[0]]
    out = []
    for idx, mu in enumerate(weights[1:], 1):
        t1 = time.time()
        lseq, rseq = sequences_by_weight[mu]
        degree = sum(Gr.crossing_grading(lseq[i], lseq[j]) for j in range(len(lseq)) for i in range(j))
        degree += sum(Gr.crossing_grading(lseq[i], left_framing) for i in range(len(lseq)))
        shift_h = -len(lseq) + 1
        extra = mono_degree(resc[idx]) - mono_degree(resc[0])
        shifted = stab[mu][shift_h, -degree + sign * extra]
        hom_set = shifted.hom_set(e_brane)
        ext_set = hom_set.homology()
        ext_basis = ext_set.basis
        n_ext = len(ext_basis)
        ambient = len(hom_set.basis._ambient_homset.basis)
        cycles = len(hom_set.basis)
        boundaries = ext_basis._coordinate_boundary_module.rank()
        print("step {}: S[{}]  graded maps {}, chain maps {}, null-homotopic {}, Ext {}  [{:.1f}s]".format(
            idx, mu, ambient, cycles, boundaries, n_ext, time.time() - t1), flush=True)
        out.append((ambient, cycles, boundaries, n_ext))
        if n_ext == 0:
            print("   no class at this shift")
            return out, None
        rep = ext_basis(0).representative()
        e_brane = rep.cone()
        _ = e_brane.differential
    return out, e_brane


if __name__ == "__main__":
    case = sys.argv[1]
    sign = -1 if (len(sys.argv) > 2 and sys.argv[2] == "-") else +1
    run_poly(case, sign)
