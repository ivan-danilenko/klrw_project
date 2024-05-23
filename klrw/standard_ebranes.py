from sage.combinat.root_system.root_system import RootSystem
from sage.combinat.root_system.weight_space import WeightSpaceElement
from sage.combinat.posets.posets import Poset
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from klrw.framed_dynkin import (
    FramedDynkinDiagram_with_dimensions,
    NodeInFramedQuiver,
)
from klrw.klrw_algebra import KLRWAlgebra
from klrw.perfect_complex import KLRWHomOfPerfectComplexes
from klrw.stable_envelopes import stable_envelope


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


def index_of_a_simple_root(root: WeightSpaceElement, Phi: RootSystem):
    root = Phi.weight_space()(root)
    for ind, al in Phi.weight_space().simple_roots().items():
        if al == root:
            return ind
    else:
        raise ValueError("Weight {} not a simple root".format(root))


def standard_ebranes(Phi: RootSystem, lambd: WeightSpaceElement):
    CT = Phi.cartan_type()

    # checking if lambd is a weight of Phi
    assert Phi.weight_space().has_coerce_map_from(
        lambd.parent()
    ), "The weight {} is not a weight for {}".format(lambd, Phi)

    # checking if lambd is proportional to fundamental
    assert len(lambd) == 1, "The weight {} is not fundamental".format(lambd)

    ((lam_index, scalar),) = tuple(lambd)

    assert scalar == 1, "The weight {} is not fundamental".format(lambd)
    # finding the index of the weight
    for i in CT.index_set():
        if lambd == Phi.weight_space().fundamental_weight(i):
            lam_index = i
            break
    else:
        "The weight {} is not fundamental for {}".format(lambd, Phi)

    # checking if the weight is minuscule
    allowed_values = set((1, 0, -1))
    assert all(
        lambd.scalar(al) in allowed_values for al in Phi.coroot_lattice().roots()
    ), "The weight {} is not minuscule".format(lambd)

    print("Finding weight poset")
    Orb = lambd.orbit()

    def leq(x1, x2, Phi=Phi):
        x = Phi.ambient_space()(x2 - x1)
        ys = [
            Phi.coambient_space()(y).coerce_to_sl()
            for y in Phi.coweight_lattice().fundamental_weights()
        ]

        if Phi.cartan_type().type() == "A":
            ys = [y.coerce_to_sl() for y in ys]
        if Phi.cartan_type().type() == "E":
            if Phi.cartan_type().rank() == 6:
                ys = [y.coerce_to_e6() for y in ys]
            elif Phi.cartan_type().rank() == 7:
                ys = [y.coerce_to_e7() for y in ys]

        return all(x.inner_product(y) >= 0 for y in ys)

    pos = Poset(data=(Orb, leq))
    pos.plot(
        label_elements=False, title="Poset of the Weight Orbit", fontsize=10
    ).show()
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
        right_sequence = [
            index_of_a_simple_root(
                right_weight_sequence[i] - right_weight_sequence[i + 1], Phi
            )
            for i in range(len(right_weight_sequence) - 1)
        ]
        right_sequence = tuple(NodeInFramedQuiver(r) for r in right_sequence)
        # take arbitrary maximal chain ends with mu
        # [i.e. starts with my in the dual poset]
        left_weight_sequence = next(
            pos.maximal_chains_iterator(
                [
                    mu,
                ]
            )
        )
        left_sequence = [
            index_of_a_simple_root(
                left_weight_sequence[i] - left_weight_sequence[i - 1], Phi
            )
            for i in range(len(left_weight_sequence) - 1, 0, -1)
        ]
        left_sequence = tuple(NodeInFramedQuiver(r) for r in left_sequence)
        sequences_by_weight[mu] = (left_sequence, right_sequence)

    left_framing = NodeInFramedQuiver(lam_index, framing=True)
    dual_lam_index = CT.opposition_automorphism()[lam_index]
    right_framing = NodeInFramedQuiver(dual_lam_index, framing=True)

    print("Defining KLRW algebra")
    # define the corresponding framed quiver
    DD = FramedDynkinDiagram_with_dimensions(CT)
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
        edge_scaling=True,
        vertex_scaling=True,
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

    R = QQ
    KLRW_one_grading = KLRWAlgebra(
        R,
        DD,
        default_vertex_parameter=R.one(),
        default_edge_parameter=R.one(),
        warnings=True,
    )

    stab = {mu: st.base_change(KLRW_one_grading) for mu, st in stab.items()}

    print("Finding the e-brane")

    weights = pos_dual.linear_extension()
    print("Started with the stable envelope for {}".format(mu))
    e_brane = stab[weights[0]]

    for mu in weights[1:]:
        print("Adding the stable envelope for {}".format(mu))
        left_seq, right_seq = sequences_by_weight[mu]
        stab_mu = stab[mu]
        degree = sum(
            KLRW_one_grading.grading_group.crossing_grading(left_seq[i], left_seq[j])
            for j in range(len(left_seq))
            for i in range(j)
        )
        degree += sum(
            KLRW_one_grading.grading_group.crossing_grading(left_seq[i], left_framing)
            for i in range(len(left_seq))
        )
        rhom = KLRWHomOfPerfectComplexes(stab_mu[-len(left_seq) + 1, -degree], e_brane)
        ext = rhom.ext(0)
        v = ext.basis()[0]
        e_brane = v.cone()

    print(e_brane)

    left_seq, right_seq = next(iter(sequences_by_weight.values()))
    relevant_state = KLRW_one_grading.state(
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
