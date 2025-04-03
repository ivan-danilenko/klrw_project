from functools import cache
from typing import Iterator

from sage.combinat.root_system.root_system import RootSystem
from sage.combinat.root_system.weight_space import WeightSpaceElement
from sage.combinat.posets.posets import Poset
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.matrix.constructor import matrix

from klrw.framed_dynkin import (
    FramedDynkinDiagram_with_dimensions,
    NodeInFramedQuiver,
)
from klrw.klrw_algebra import KLRWAlgebra
from klrw.stable_envelopes import stable_envelope
from klrw.perfect_complex import KLRWPerfectComplex


def minuscule_weights(Phi: RootSystem):
    minuscule_weights = []
    allowed_values = set((1, 0, -1))
    coroots = Phi.coroot_lattice().roots()
    minuscule_weights = tuple(
        x
        for x in Phi.weight_lattice().fundamental_weights()
        if all(x.scalar(y) in allowed_values for y in coroots)
    )
    return minuscule_weights


@cache
def weight_poset_in_minuscule_rep(Phi: RootSystem, lambd: WeightSpaceElement):
    # checking if lambd is a weight of Phi
    assert Phi.weight_space().has_coerce_map_from(
        lambd.parent()
    ), "The weight {} is not a weight for {}".format(lambd, Phi)

    # checking if lambd is proportional to fundamental
    assert len(lambd) == 1, "The weight {} is not fundamental".format(lambd)

    # checking if the weight is minuscule
    allowed_values = set((1, 0, -1))
    assert all(
        lambd.scalar(al) in allowed_values for al in Phi.coroot_lattice().roots()
    ), "The weight {} is not minuscule".format(lambd)

    print("Finding weight poset")
    orb = lambd.orbit()
    rels = [
        (weight, weight + Phi.root_lattice().simple_root(simple_root_index))
        for weight in orb
        for simple_root_index in weight.descents()
    ]
    return Poset((orb, rels), cover_relations=True)


def moving_strands_from_weights(weight_sequence: Iterator):
    """
    Given a sequence `mu_0, ... ,mu_k`
    where `mu_m - mu_{m-1}` is a simple root (call it `alpha_{i(m)}`),
    generates sequence `(Vi(1), ..., Vi(k)`.
    """
    previous_weight = next(weight_sequence)
    for next_weight in weight_sequence:
        index = (previous_weight - next_weight).to_simple_root()
        yield NodeInFramedQuiver(index)
        previous_weight = next_weight


@cache
def standard_reduced_ebranes(Phi: RootSystem, lambd: WeightSpaceElement):
    # TODO: rework, merging common parts with `standard_ebranes`.
    # make klrw_options the same.
    from klrw.one_term_complex import one_term_complex

    ((lam_index, scalar),) = tuple(lambd)
    # finding the index of the weight
    for i in Phi.cartan_type().index_set():
        if lambd == Phi.weight_space().fundamental_weight(i):
            lam_index = i
            break
    else:
        "The weight {} is not fundamental for {}".format(lambd, Phi)

    left_framing = NodeInFramedQuiver(lam_index, framing=True)
    dual_lam_index = Phi.cartan_type().opposition_automorphism()[lam_index]
    right_framing = NodeInFramedQuiver(dual_lam_index, framing=True)

    klrw_options = {
        "dot_scaling": True,
        "edge_scaling": True,
        "vertex_scaling": True,
        "invertible_parameters": True,
        "warnings": True,
    }

    return one_term_complex(
        base_R=ZZ,
        quiver=Phi.cartan_type(),
        state=(left_framing, right_framing),
        **klrw_options,
    )


@cache
def standard_ebranes(Phi: RootSystem, lambd: WeightSpaceElement):
    ((lam_index, scalar),) = tuple(lambd)

    assert scalar == 1, "The weight {} is not fundamental".format(lambd)
    # finding the index of the weight
    # for i in Phi.cartan_type().index_set():
    #    if lambd == Phi.weight_space().fundamental_weight(i):
    #        lam_index = i
    #        break
    # else:
    #    "The weight {} is not fundamental for {}".format(lambd, Phi)

    pos = weight_poset_in_minuscule_rep(Phi, lambd)
    # pos.plot(
    #     label_elements=False, title="Poset of the Weight Orbit", fontsize=10
    # ).show()
    pos_dual = pos.dual()

    sequences_by_weight = {}

    for mu in pos:
        # take arbitrary maximal chain starting with mu
        right_weight_sequence = next(
            pos_dual.maximal_chains_iterator(
                [
                    mu,
                ]
            )
        )
        right_sequence = tuple(moving_strands_from_weights(iter(right_weight_sequence)))
        # take arbitrary maximal chain ends with mu
        # [i.e. starts with my in the dual poset]
        left_weight_sequence = next(
            pos.maximal_chains_iterator(
                [
                    mu,
                ]
            )
        )
        left_sequence = tuple(
            moving_strands_from_weights(reversed(left_weight_sequence))
        )
        sequences_by_weight[mu] = (left_sequence, right_sequence)

    left_framing = NodeInFramedQuiver(lam_index, framing=True)
    dual_lam_index = Phi.cartan_type().opposition_automorphism()[lam_index]
    right_framing = NodeInFramedQuiver(dual_lam_index, framing=True)

    print("Defining KLRW algebra")
    # define the corresponding framed quiver
    DD = FramedDynkinDiagram_with_dimensions(Phi.cartan_type())
    DD[left_framing] += 1
    DD[right_framing] += 1
    # take the pair of sequences for any intermediate weight
    left_seq, right_seq = next(iter(sequences_by_weight.values()))
    for node in left_seq + right_seq:
        DD[node] += 1

    print("Framed quiver:")
    print(DD)

    KLRW = KLRWAlgebra(
        ZZ,
        DD,
        dot_scaling=True,
        edge_scaling=True,
        vertex_scaling=True,
        invertible_parameters=True,
        warnings=True,
    )

    print("Finding stable envlopes")
    stab = {
        mu: stable_envelope(
            KLRW,
            left_framing,
            right_framing,
            left_sequence=left_seq,
            right_sequence=right_seq,
        )
        for mu, (left_seq, right_seq) in sequences_by_weight.items()
    }

    # TODO: avoid step with going to base QQ
    # by properly implementing Ext groups over ZZ
    KLRW_QQ = KLRW._replace_(base_R=QQ)

    stab = {mu: st.base_change(KLRW_QQ) for mu, st in stab.items()}

    print("Finding the e-brane")

    weights = pos_dual.linear_extension()
    print("Started with the stable envelope for {}".format(mu))
    e_brane = stab[weights[0]]

    for mu in weights[1:]:
        print("Adding the stable envelope for {}".format(mu))
        left_seq, right_seq = sequences_by_weight[mu]
        stab_mu = stab[mu]
        degree = sum(
            KLRW_QQ.grading_group.crossing_grading(left_seq[i], left_seq[j])
            # KLRW_one_grading.grading_group.crossing_grading(left_seq[i], left_seq[j])
            for j in range(len(left_seq))
            for i in range(j)
        )
        degree += sum(
            KLRW_QQ.grading_group.crossing_grading(left_seq[i], left_framing)
            # KLRW_one_grading.grading_group.crossing_grading(left_seq[i], left_framing)
            for i in range(len(left_seq))
        )
        hom_set = stab_mu[-len(left_seq) + 1, -degree].hom_set(e_brane)
        ext_set = hom_set.homology()
        ext_basis = ext_set.basis
        assert len(ext_basis) == 1
        ext = ext_basis(0)
        e_brane = ext.cone()

    # going back to ZZ base
    differential_ZZ_dict = {}
    for hom_deg, mat in e_brane.differential:
        mat_dict_ZZ = {
            (i, j): KLRW.sum_of_terms(
                (braid, KLRW.base()(coeff)) for braid, coeff in entry.value
            )
            for (i, j), entry in mat.dict(copy=False).items()
        }
        differential_ZZ_dict[hom_deg] = matrix(
            ring=KLRW.opposite,
            ncols=mat.ncols(),
            nrows=mat.nrows(),
            entries=mat_dict_ZZ,
        )

    e_brane = KLRWPerfectComplex(
        ring=KLRW,
        projectives=e_brane.projectives(),
        differential=differential_ZZ_dict,
        differential_degree=e_brane.differential.degree(),
    )
    return e_brane

    left_seq, right_seq = next(iter(sequences_by_weight.values()))
    relevant_state = KLRW_QQ.state(
        (left_framing,) + left_seq + right_seq + (right_framing,)
    )

    #########
    # Warning: if weight poset is not linear, then there are several states that
    # correspond to the same simple module. So far it won't work correctly
    ########
    if not pos.is_chain():
        raise NotImplementedError("To compute unknot the weight poset must be linear")

    rhom_complex = e_brane.rhom_to_simple(relevant_state, dualize_complex=False)

    G = rhom_complex.grading_group()
    # transformation to match standard grading in Khovanov homology
    transformation = G.hom(
        [
            G((-1, 0)),
            G((-1, -1)),
        ]
    )

    homology_raw = rhom_complex.homology(base_ring=ZZ)

    homology = {}
    for grading, homology_group in homology_raw.items():
        if homology_group.ngens() != 0:
            homology[transformation(grading)] = homology_group

    return homology
