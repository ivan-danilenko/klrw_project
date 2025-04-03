from dataclasses import dataclass, field
from types import MappingProxyType
from collections import defaultdict
import operator

from sage.combinat.root_system.weight_space import WeightSpaceElement
from sage.combinat.free_module import CombinatorialFreeModule
from sage.categories.action import Action
from sage.misc.lazy_attribute import lazy_attribute, lazy_class_attribute
from sage.structure.unique_representation import UniqueRepresentation
from sage.categories.finite_dimensional_modules_with_basis import (
    FiniteDimensionalModulesWithBasis,
)

from klrw.framed_dynkin import NodeInFramedQuiver
from klrw.klrw_algebra import KLRWAlgebra, KLRWElement


class EBrane_abstract:
    """
    An abstract class for E-branes
    """

    def cover(self):
        raise NotImplementedError("Abstract class")

    def projective_covers_iter(self):
        """
        For every projective that covers the simple
        (all of them are isomorphic, but correspond to different states)
        yields a pair
        `(state, braid)`
        where `state` is a tuple representing
        the state corresponding to the pojective,
        and `braid` is a dictionary representing the braid
        relating `state` to the fixed (in `_a_projective_cover`)
        projective cover.
        """
        raise NotImplementedError()

    def resolution(self):
        raise NotImplementedError()


@dataclass(frozen=True, init=True, eq=True, repr=False)
class StandardEBrane_base(EBrane_abstract):
    """
    An abstract class for standard E-branes
    """

    color: WeightSpaceElement
    _cover: tuple[NodeInFramedQuiver] = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        # bypassing protection `frozen=True`
        object.__setattr__(
            self,
            "_cover",
            self._a_projective_cover_(),
        )

    def cover(self):
        return self._cover

    def _a_projective_cover_(self):
        """
        Returns a state of a projective that covers this E-brane.
        It is randomly picked, we save it in `self._cover`.

        The result is a tuple, not a KLRW state.
        """
        raise NotImplementedError()

    def root_system(self):
        return self.color.parent().root_system

    def cartan_type(self):
        return self.color.parent().cartan_type()

    def projective_covers_iter(self):
        """
        For every projective that covers the simple
        (all of them are isomorphic, but correspond to different states)
        yields a pair
        `(state, braid)`
        where `state` is a tuple representing
        the state corresponding to the pojective,
        and `braid` is a dictionary representing the braid
        relating `state` to the fixed (in `_a_projective_cover`)
        projective cover.
        """
        raise NotImplementedError()

    def _color_index(self):
        ((lam_index, scalar),) = tuple(self.color)
        assert scalar == 1, "The weight {} is not fundamental".format(self.color)

        return lam_index

    def _left_framing(self):
        return NodeInFramedQuiver(self._color_index(), framing=True)

    def _right_framing(self):
        dual_index = self.cartan_type().opposition_automorphism()[self._color_index()]
        return NodeInFramedQuiver(dual_index, framing=True)

    def _state_tuple_from_moving_part(self, moving_part):
        state_tuple = (self._left_framing(),)
        state_tuple += tuple(moving_part)
        state_tuple += (self._right_framing(),)

        return state_tuple

    def resolution(self):
        raise NotImplementedError()


@dataclass(frozen=True, init=True, eq=True, repr=False)
class StandardEBrane(StandardEBrane_base):
    # color: WeightSpaceElement
    # cover: tuple[NodeInFramedQuiver] = field(init=False, compare=False, hash=False)
    poset: tuple[NodeInFramedQuiver] = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        # bypassing protection `frozen=True`
        object.__setattr__(
            self,
            "poset",
            self._poset(),
        )
        super().__post_init__()

    def __repr__(self):
        return "E_{" + repr(self.color) + "}"

    def _poset(self):
        from klrw.standard_ebranes import weight_poset_in_minuscule_rep

        return weight_poset_in_minuscule_rep(self.root_system(), self.color)

    def _a_projective_cover_(self):
        """
        Returns a state of a projective that covers this E-brane.
        It is randomly picked, we save it in `self._cover`.

        The result is a tuple, not a KLRW state.
        """
        from klrw.standard_ebranes import moving_strands_from_weights

        # picking any chain
        chain = next(self.poset.dual().maximal_chains_iterator())
        # and returning state tuple corresponding to it
        return self._state_tuple_from_moving_part(
            moving_strands_from_weights(iter(chain))
        )

    @staticmethod
    def _braid_between_projective_covers_(state_left, state_right):
        """
        Creates a braid with minimal number of intersections.

        Returns the data as a dictionary representing the braid.
        """
        braid_dict = {}
        vertices = set(state_left)
        assert set(state_right) == vertices
        for vertex in vertices:
            left_ends = (i for i, color in enumerate(state_left) if color == vertex)
            right_ends = (i for i, color in enumerate(state_right) if color == vertex)
            for i, j in zip(left_ends, right_ends, strict=True):
                braid_dict[i] = j
        return braid_dict

    def projective_covers_iter(self):
        """
        For every projective that covers the simple
        (all of them are isomorphic, but correspond to different states)
        yields a pair
        `(state, braid)`
        where `state` is a tuple representing
        the state corresponding to the pojective,
        and `braid` is a dictionary representing the braid
        relating `state` to the fixed (in `_a_projective_cover`)
        projective cover.
        """
        from klrw.standard_ebranes import moving_strands_from_weights

        fixed_state_tuple = self.cover()
        for chain in self.poset.dual().maximal_chains_iterator():
            state_tuple = self._state_tuple_from_moving_part(
                moving_strands_from_weights(iter(chain))
            )
            braid = self._braid_between_projective_covers_(
                state_tuple, fixed_state_tuple
            )
            yield (state_tuple, braid)

    def resolution(self):
        from klrw.standard_ebranes import standard_ebranes

        return standard_ebranes(self.root_system(), self.color)


class StandardReducedEBrane(StandardEBrane_base):
    def __repr__(self):
        return "E_{" + repr(self.color) + "}_red"

    def _a_projective_cover_(self):
        """
        Returns a state of a projective that covers this E-brane.
        It is randomly picked, we save it in `self._cover`.

        The result is a tuple, not a KLRW state.
        """
        # Returning the state tuple with no moving strands
        return self._state_tuple_from_moving_part([])

    @staticmethod
    def _braid_between_projective_covers_(state_left, state_right):
        """
        Creates a braid with minimal number of intersections.

        Returns the data as a dictionary representing the braid.
        """
        braid_dict = {}
        assert state_right == state_left
        for i in range(len(state_right)):
            braid_dict[i] = i
        return braid_dict

    def projective_covers_iter(self):
        """
        For every projective that covers the simple
        (all of them are isomorphic, but correspond to different states)
        yields a pair
        `(state, braid)`
        where `state` is a tuple representing
        the state corresponding to the pojective,
        and `braid` is a dictionary representing the braid
        relating `state` to the fixed (in `_a_projective_cover`)
        projective cover.
        """
        fixed_state_tuple = self.cover()
        state_tuple = self._state_tuple_from_moving_part([])
        braid = self._braid_between_projective_covers_(state_tuple, fixed_state_tuple)
        yield (state_tuple, braid)

    def resolution(self):
        from klrw.standard_ebranes import standard_reduced_ebranes

        return standard_reduced_ebranes(self.root_system(), self.color)


@dataclass(frozen=True, init=False, eq=True, repr=False)
class EBrane(EBrane_abstract):
    standard_pieces: tuple[StandardEBrane]

    def __init__(self, *colors, reduced_parts=set()):
        assert colors, "At least one color has to be given."
        standard_pieces = tuple(
            StandardReducedEBrane(col) if i in reduced_parts else StandardEBrane(col)
            for i, col in enumerate(colors)
        )
        # bypassing protection `frozen=True`
        object.__setattr__(
            self,
            "standard_pieces",
            standard_pieces,
        )

    @lazy_attribute
    def root_system(self):
        from klrw.misc import get_from_all_and_assert_equality

        return get_from_all_and_assert_equality(
            lambda piece: piece.root_system(), self.standard_pieces
        )

    def cover(self):
        result = ()
        for piece in self.standard_pieces:
            result += piece.cover()

        return result

    @staticmethod
    def _merge_states_(states):
        result = ()
        for piece in states:
            result += piece.cover()

        return result

    def projective_covers_iter(self):
        from itertools import product

        covers_iter = product(
            *(piece.projective_covers_iter() for piece in self.standard_pieces)
        )
        for covers in covers_iter:
            states, braids = zip(*covers)
            new_state = ()
            new_braid = {}
            for br, st in zip(braids, states):
                for i, j in br.items():
                    new_braid[i + len(new_state)] = j + len(new_state)
                new_state += st

            yield (new_state, new_braid)

    def __repr__(self):
        return " @ ".join(repr(piece) for piece in self.standard_pieces)

    def resolution(self):
        # prepare pieces
        pieces_set = set(self.standard_pieces)
        resolution_pieces = {piece: piece.resolution() for piece in pieces_set}
        # first make over ZZ.
        pieces_iter = iter(self.standard_pieces)
        piece = next(pieces_iter)
        resolution = resolution_pieces[piece]
        # making tensor products
        for piece in pieces_iter:
            resolution @= resolution_pieces[piece]

        return resolution


class KLRWLeftActionOnEBrane(Action):
    def _act_(
        self,
        q: KLRWElement,
        x: KLRWElement,
    ) -> KLRWElement:
        product = q * self.domain()._lift_(x)
        return self.codomain()._project_element_(product)


class KLRWProjectivesToEBrane_Homset(CombinatorialFreeModule):
    """
    The direct sum of homsets `KLRWProjective -> EBrane`
    over KLRWAlgebra.

    This is naturally a left module over KLRW Algebra.
    Over the subring of parameters, this is a finite
    rank module.

    Elements are presented by preferred lifts
    to the KLRW algebra.
    """

    Element = KLRWElement

    def __classcall__(
        cls,
        ring: KLRWAlgebra,
        ebrane_data: EBrane_abstract | WeightSpaceElement,
    ):
        if isinstance(ebrane_data, WeightSpaceElement):
            ebrane_data = StandardEBrane(ebrane_data)
        if not isinstance(ebrane_data, EBrane_abstract):
            raise ValueError(
                "Unacceptable type of simple data: {}".format(ebrane_data.__class__)
            )

        return super().__classcall__(
            cls,
            ring=ring,
            ebrane_data=ebrane_data,
        )

    def __init__(
        self,
        ring: KLRWAlgebra,
        ebrane_data: StandardEBrane,
    ):
        self._ebrane = ebrane_data
        self._klrw_algebra = ring
        dot_algebra = self._klrw_algebra.base().without_dots

        category = FiniteDimensionalModulesWithBasis(dot_algebra)
        # note: with strands of different color not allowed to cross in the
        # basis braids
        CombinatorialFreeModule.__init__(
            self,
            R=dot_algebra,
            element_class=self.Element,
            category=category,
        )

    def KLRW_algebra(self):
        return self._klrw_algebra

    def cover(self):
        """
        Returns a state of a projective that covers this E-brane.
        It is randomly picked, but stays the same for the instance.
        """
        return self.KLRW_algebra().state(self._ebrane.cover())

    def _basis_braids_iter_(self):
        right_state = self.cover()
        braid_set = self.KLRW_algebra().braid_set()
        for left_state_tuple, braid_dict in self._ebrane.projective_covers_iter():
            left_state = self.KLRW_algebra().state(left_state_tuple)
            braid = braid_set.braid_by_extending_permutation(right_state, braid_dict)
            yield left_state, braid

    @lazy_attribute
    def _basis_braids_(self):
        result = {left_state: braid for left_state, braid in self._basis_braids_iter_()}

        return MappingProxyType(result)

    @lazy_attribute
    def basis(self):
        """
        Returns a basis over the parameter algebra.
        """
        result = {
            left_state: self.monomial(braid)
            for left_state, braid in self._basis_braids_iter_()
        }

        return MappingProxyType(result)

    @lazy_attribute
    def _relevant_braids_(self):
        return frozenset(braid for _, braid in self._basis_braids_iter_())

    def _project_element_(self, element: KLRWElement):
        """
        Bring `element` to standard form.

        The module is a quotient
        `Hom(SumOfProjectives, ProjectiveCover) -> Hom(SumOfProjectives, EBrane)`
        We present elements as lifts to the first Hom, i.e. image
        along a preferred linear map
        `Hom(SumOfProjectives, EBrane) -> Hom(SumOfProjectives, ProjectiveCover)`
        (not a map of modules).

        Given an `element` of KLRW Algebra, we project it along
        `KLRWAlgebra -> Hom(SumOfProjectives, ProjectiveCover)`
        then along
        `Hom(SumOfProjectives, ProjectiveCover) -> Hom(SumOfProjectives, EBrane)`,
        then lift along
        `Hom(SumOfProjectives, EBrane) -> Hom(SumOfProjectives, ProjectiveCover)`.
        """
        quotient_by_dots_morphism = self.KLRW_algebra().base().quotient_by_dots
        projection_dict = {
            braid: new_coeff
            for braid, coeff in element
            if braid in self._relevant_braids_
            and (new_coeff := quotient_by_dots_morphism(coeff))
        }
        return self._from_dict(projection_dict, coerce=False, remove_zeros=False)

    def zero(self):
        return self._from_dict({})

    def _lift_(self, element: KLRWElement):
        return self.KLRW_algebra()._from_dict(
            element.monomial_coefficients(copy=True), coerce=False, remove_zeros=False
        )

    def _get_action_(self, other, op, self_on_left):
        if op == operator.mul:
            if not self_on_left:
                if isinstance(other, KLRWAlgebra):
                    return KLRWLeftActionOnEBrane(G=other, S=self)

        return super()._get_action_(other, op, self_on_left)

    def _functor_on_morphism_betweem_projectives_(self, element: KLRWElement):
        """
        Given an `element` of KLRW Algebra, we
        concider it as a map between projectives,
        i.e. an element of
        `Hom(P_l, P_r)`,
        where `P_l` is the projective corresponding to the left end,
        `P_r` is the projective corresponding to the right end.

        We return a corresponding map
        `Hom(P_r[shift], EBrane) -> Hom(P_l[shift], EBrane)`.
        There is at most one `shift` where each module is non-zero.
        These are rank one modules over the algebra of parameters,
        so we return an element of the parameter algebra
        with respect to standard basis in both modules.
        If for all shifts the modules are trivial, we return zero.
        """
        left_state = element.left_state(check_if_all_have_same_left_state=True)
        right_state = element.right_state(check_if_all_have_same_right_state=True)

        if left_state not in self.basis:
            return self.base().zero()
        domain_basis_vector = self.basis[left_state]
        if right_state not in self._basis_braids_:
            return self.base().zero()
        codomain_basis_braid = self._basis_braids_[right_state]
        image = element * domain_basis_vector
        matrix_element = image[codomain_basis_braid]

        return matrix_element

    def _repr_(self):
        return "Homset from the KLRW algebra to " + repr(self._ebrane)

    def _replace_(self, **replacements):
        """
        Make a similar parent with several adjustments.

        Compare to _replace of named tuples.
        """
        from sage.structure.unique_representation import unreduce

        cls, args, kwrds = self._reduction
        new_kwrds = kwrds | replacements
        return unreduce(cls, args, new_kwrds)

    def _repr_term(self, monomial):
        """
        Single-underscore method for string representation of basis elements
        """

        s = "E_" + monomial.left_state().__repr__()
        for i in monomial.word():
            s += "*s{}".format(i)

        if monomial.word():
            s += "*E_" + monomial.right_state().__repr__()

        return s


class KLRWPerfectComplexToEbrane_Homset(UniqueRepresentation):
    def __init__(self, perfect_complex, ebrane):
        self._perfect_complex = perfect_complex
        self._ebrane = ebrane
        self.projectives_to_ebrane_homset = KLRWProjectivesToEBrane_Homset(
            self._perfect_complex.KLRW_algebra(), self._ebrane
        )

    @lazy_attribute
    def _relevant_indices_(self):
        relevant_indices = defaultdict(dict)
        for hom_deg, projs in self._perfect_complex.projectives_iter():
            for index, pr in enumerate(projs):
                if pr.state in self.projectives_to_ebrane_homset.basis:
                    eq_deg = pr.equivariant_degree.ordinary_grading(as_scalar=False)
                    # since we take RHom(-,E) degree becomes opposite
                    deg = -self.to_standard_grading(hom_deg, eq_deg)
                    new_index = len(relevant_indices[deg])
                    relevant_indices[deg][index] = new_index

        return MappingProxyType(relevant_indices)

    @lazy_class_attribute
    def standard_grading_group(cls):
        from sage.groups.additive_abelian.additive_abelian_group import (
            AdditiveAbelianGroup,
        )

        return AdditiveAbelianGroup([0, 0])

    @classmethod
    def to_standard_grading(cls, hom_grading, eq_grading):
        """
        Translates KLRW grading into standard grading for link homology.
        """
        hom_label = hom_grading.parent().homological_grading_label
        hom_index = hom_grading.coefficient(hom_label)
        eq_index = eq_grading.ordinary_grading(as_scalar=True)

        return cls.standard_grading_group((-hom_index - eq_index, -eq_index))

    @lazy_attribute
    def standard_differential_degree(self):
        # since we take RHom(-,E) degree of the differential
        # becomes opposite
        diff_degree = -self._perfect_complex.differential.degree()
        standard_diff_degree = self.to_standard_grading(
            diff_degree.homological_part(),
            diff_degree.equivariant_part().ordinary_grading(as_scalar=False),
        )

        return standard_diff_degree

    def _differentials_(self):
        from sage.matrix.constructor import matrix

        scalars = self._perfect_complex.KLRW_algebra().scalars()

        differential = {}
        for deg in self._relevant_indices_:
            codomain_degrees = deg - self.standard_differential_degree

            if codomain_degrees not in self._relevant_indices_:
                ncols = 0
            else:
                ncols = len(self._relevant_indices_[codomain_degrees])

            nrows = len(self._relevant_indices_[deg])

            differential[deg] = matrix(
                scalars,
                nrows=nrows,
                ncols=ncols,
                sparse=True,
            )

        homset = self.projectives_to_ebrane_homset
        parameter_algebra = homset.base()
        functor = homset._functor_on_morphism_betweem_projectives_
        for hom_deg, diff in self._perfect_complex.differential:
            for (i, j), elem in diff.dict(copy=False).items():
                new_elem = functor(elem.value)
                new_elem = parameter_algebra.hom_to_simple(new_elem)
                if not new_elem.is_zero():
                    left_proj = self._perfect_complex.projectives(hom_deg)[j]
                    eq_deg = left_proj.equivariant_degree
                    # since we take RHom(-,E) gradings become opposite
                    # and domain and codomains swap
                    # it gives an extra degree shift
                    codom_deg = -self.to_standard_grading(hom_deg, eq_deg)
                    dom_deg = codom_deg + self.standard_differential_degree
                    i_new = self._relevant_indices_[codom_deg][j]
                    j_new = self._relevant_indices_[dom_deg][i]

                    grading = dom_deg

                    differential[grading][j_new, i_new] = new_elem

        return differential

    def chain_complex(self, base_ring=None):
        from sage.homology.chain_complex import ChainComplex

        return ChainComplex(
            data=self._differentials_(),
            degree_of_differential=self.standard_differential_degree,
            grading_group=self.standard_grading_group,
        )

    def homology(self, base_ring=None):
        return self.chain_complex().homology(base_ring=base_ring)
