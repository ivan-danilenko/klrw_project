from itertools import product
from collections.abc import Iterable
from dataclasses import dataclass
from copy import deepcopy

from types import MappingProxyType
import operator

from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.categories.finite_dimensional_algebras_with_basis import (
    FiniteDimensionalAlgebrasWithBasis,
)
from sage.categories.action import Action
from sage.rings.polynomial.polydict import ETuple
from sage.misc.lazy_attribute import lazy_attribute
from sage.misc.cachefunc import cached_method
from sage.structure.unique_representation import UniqueRepresentation
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector

from .klrw_state import KLRWstate
from .klrw_braid import KLRWbraid, KLRWbraid_set
from .framed_dynkin import (
    FramedDynkinDiagram_with_dimensions,
    KLRWUpstairsDotsAlgebra,
)
from .bimodule_monoid import LeftFreeBimoduleMonoid

from .klrw_endomorphism import (
    KLRWEndomorphismAlgebra,
    LeftKLRWEndomorphismAction,
    RightKLRWEndomorphismAction,
)


class KLRWElement(IndexedFreeModuleElement):
    def degree(self, check_if_homogeneous=False):
        # zero elements return None as degree
        degree = None
        for braid, coeff in self:
            if not coeff.is_zero():
                braid_degree = self.parent().braid_degree(braid)
                coeff_degree = coeff.parent().element_degree(
                    coeff, check_if_homogeneous=check_if_homogeneous
                )
                term_degree = braid_degree + coeff_degree

                if not check_if_homogeneous:
                    return term_degree
                elif degree is None:
                    degree = term_degree
                elif degree != term_degree:
                    raise ValueError("The KLRW element is not homogeneous!")

        return degree

    def right_state(self, check_if_all_have_same_right_state=False):
        # raises error if terms have different right states
        # and check_if_all_have_same_right_state=True
        # returns None for zero elements
        right_state = None
        for braid, coeff in self:
            if not coeff.is_zero():
                braid_right_state = braid.right_state()
                if not check_if_all_have_same_right_state:
                    return braid_right_state
                elif right_state is None:
                    right_state = braid_right_state
                elif right_state != braid_right_state:
                    raise ValueError("The KLRW element has different right states!")

        return right_state

    def left_state(self, check_if_all_have_same_left_state=False):
        # raises error if terms have different left states
        # and check_if_all_have_same_left_state=True
        # returns None for zero elements
        left_state = None
        for braid, coeff in self:
            if not coeff.is_zero():
                braid_left_state = braid.left_state()
                if not check_if_all_have_same_left_state:
                    return braid_left_state
                elif left_state is None:
                    left_state = braid_left_state
                elif left_state != braid_left_state:
                    raise ValueError("The KLRW element has different left states!")

        return left_state

    def as_matrix_in_graded_component(self, graded_component, acting_on_left=True):
        """
        Gives a matrix of how multiplication by this element
        acts on a graded component of a Hom between projectives.
        Return None if the element is zero, since we don't know
        the codomain.
        """
        assert graded_component.KLRW_algebra is self.parent()
        # use to assert that the element is homogenuous in degree
        self.degree(check_if_homogeneous=True)
        # use to assert that the element has the same left state in all terms
        self.left_state(check_if_all_have_same_left_state=True)
        # use to assert that the element has the same right state in all terms
        self.right_state(check_if_all_have_same_right_state=True)

        return sum(
            graded_component.basis_as_matrix_in_graded_component(
                braid,
                coeff,
                acting_on_left=acting_on_left,
            )
            for braid, coeff in self
        )

    # TODO:rewrite
    # def check(self):

    # TODO: add checks that all braids are correct


class RightDotAction(Action):
    # @cached_method
    def _act_(self, p, x: IndexedFreeModuleElement) -> IndexedFreeModuleElement:
        return self.codomain().linear_combination(
            (self._act_on_bases_(exp_tuple, braid), coeff * left_poly)
            for exp_tuple, coeff in p.iterator_exp_coeff(as_ETuples=True)
            for braid, left_poly in x
        )

    @cached_method
    def _act_on_bases_(self, p_exp_tuple: ETuple, braid: KLRWbraid):
        return self.codomain().sum(self._act_on_bases_iter_(p_exp_tuple, braid))

    def _act_on_bases_iter_(self, p_exp_tuple: ETuple, braid: KLRWbraid):
        # idempotents commute with dots
        if len(braid.word()) == 0:
            yield self.codomain().term(
                braid,
                self.actor().monomial(*p_exp_tuple)
            )
        else:
            last_letter = braid.word()[-1]
            c1, c2 = braid.intersection_colors(-1)
            if c1 == c2:
                index_of_dots_on_left_strand = (
                    self.actor()
                    .variables[
                        c1, braid.right_state().index_among_same_color(last_letter - 1)
                    ]
                    .position
                )
                index_of_special_coefficient = self.actor().variables[c1].position
                # terms where crossing stays
                coeff_of_crossing = self._coeff_of_crossing_(
                    p_exp_tuple, index_of_dots_on_left_strand
                )
                for exp_tuple, coeff in coeff_of_crossing.iterator_exp_coeff(
                    as_ETuples=True
                ):
                    for term in self._act_on_bases_iter_(exp_tuple, braid[:-1]):
                        for br, poly in term:
                            # we use :meth:right_multiply_by_s, to bring to lexmin form
                            yield coeff * poly * self.codomain().right_multiply_by_s(
                                br, braid[-1]
                            )
                # terms where crossing goes away
                for poly in self._coeff_of_correction_(
                    p_exp_tuple,
                    index_of_dots_on_left_strand,
                    index_of_special_coefficient,
                ):
                    for exp_tuple, coeff in poly.iterator_exp_coeff(as_ETuples=True):
                        yield coeff * self._act_on_bases_(exp_tuple, braid[:-1])

            else:
                for term in self._act_on_bases_iter_(p_exp_tuple, braid[:-1]):
                    for br, poly in term:
                        yield poly * self.codomain().right_multiply_by_s(br, braid[-1])

    def _coeff_of_crossing_(self, p_exp_tuple: ETuple, i: int):
        """
        We will write X*dot = _coeff_of_crossing_*X + _coeff_of_correction_*||
        Here X is a simple crossing and || is a smoothed version
        i and i+1 are the indices of variables corresponding
        to the dots on the crossing strands
        """
        new_exp = list(p_exp_tuple)
        new_exp[i + 1], new_exp[i] = new_exp[i], new_exp[i + 1]
        return self.actor().monomial(*new_exp)

    def _coeff_of_correction_(self, p_exp_tuple: ETuple, i: int, j: int) -> Iterable:
        """
        We will write X*dot = _coeff_of_crossing_*X + r*_coeff_of_correction_*||
        Here X is a simple crossing and || is a smoothed version
        i and i+1 are the indices of variables corresponding to the dots
        on the crossing strands
        j is the index of r
        """
        # returns None if two degrees coincide [no corrections]
        if p_exp_tuple[i] != p_exp_tuple[i + 1]:
            new_exp = list(p_exp_tuple)

            if p_exp_tuple[i] > p_exp_tuple[i + 1]:
                end_degree_at_i = new_exp[i + 1] - 1
                new_exp[i] += -1
                delta = 1
            else:
                end_degree_at_i = new_exp[i + 1]
                new_exp[i + 1] += -1
                delta = -1

            # multiplying by t_{...}
            if j is None:
                # if we use default edge parameters
                coeff = self.actor().base()(delta) * self.actor().default_edge_parameter
            else:
                coeff = self.actor().base()(delta)
                new_exp[j] += 1

            while new_exp[i] != end_degree_at_i:
                # we do tuple() to make a copy
                # and plug it in later into :meth:G.monomial
                yield coeff * self.actor().monomial(*tuple(new_exp))
                new_exp[i] += -delta
                new_exp[i + 1] += delta


class KLRWAlgebra(LeftFreeBimoduleMonoid):
    # Element = KLRWElement

    def __init__(
        self,
        base_R,
        quiver: FramedDynkinDiagram_with_dimensions,
        warnings=False,
        **prefixes
    ):
        """
        Makes a KLRW algebra for a quiver.
        base_R is the base ring.
        if warnings, then extra warnings can be printed
        """
        self.warnings = warnings
        self.quiver = deepcopy(quiver)  # .copy()
        self.KLRWBraid = KLRWbraid_set(self.quiver, state_on_right=True)
        dots_algebra = KLRWUpstairsDotsAlgebra(base_R, self.quiver, **prefixes)
        category = FiniteDimensionalAlgebrasWithBasis(dots_algebra)
        super().__init__(R=dots_algebra, element_class=KLRWElement, category=category)

    def __getitem__(self, key: slice):
        """
        For an instance KLRW of KLRW algebra
        KLRW[left_state:right_state:degree] is
        the graded finite-dimensional component
        """
        if isinstance(key, slice):
            return KLRWAlgebraGradedComponent(self, key.start, key.stop, key.step)

    @lazy_attribute
    def ideal_of_symmetric_dots(self):
        return list(self.base().symmetric_dots_gens()) * self.base()

    def modulo_symmetric_dots(self, element):
        return element.reduce(self.ideal_of_symmetric_dots)

    def one(self):
        if self.warnings:
            print(
                r"The identity of KLRW is used' often this is done deep in Sage's code."
                +
                "\n"
                +
                r"It's faster to replace this by a function that does not use it."
            )
        return self._one_()

    # We don't call it :meth:one to avoid unintentional coercion from
    # the ring of dots/scalars. It works, but it too slow.
    @cached_method
    def _one_(self):
        return self.sum_of_terms(
            (self.KLRWBraid._element_constructor_(state), self.base().one())
            for state in self.KLRWBraid.KLRWstate_set
        )

    def scalars(self):
        return self.base().base()

    def gens(self):
        # dots and other coefficients
        for state in self.KLRWBraid.KLRWstate_set:
            for scalar in self.base().gens():
                yield self.term(
                    self.KLRWBraid._element_constructor_(state),
                    scalar
                )
        # simple crossings
        yield from self.gens_over_dots()

    def gens_over_dots(self):
        # simple crossings
        for state in self.KLRWBraid.KLRWstate_set:
            for i in range(1, len(state)):
                if not state[i - 1].is_framing() and not state[i].is_framing():
                    yield self.term(
                        self.KLRWBraid._element_constructor_(
                            state, (i,)
                        ),
                        self.base().one()
                    )

    def center_gens(self):
        yield from self.base().center_gens()

    def basis_in_dots_modulo_center(self):
        yield from self.base().basis_modulo_symmetric_dots()

    def basis_over_dots_and_center(self):
        """
        Returns a basis [as a left module] over the ring generated
        by dots and parameters.
        """
        for braid in self.KLRWBraid:
            yield self.monomial(braid)

    def basis_over_center(self):
        for braid, dot_poly in product(
            self.KLRWBraid, self.basis_in_dots_modulo_center()
        ):
            yield self.term(braid, dot_poly)

    def braid_degree(self, braid):
        degree = 0
        current_state = braid.right_state()

        for i in reversed(braid.word()):
            degree += -self.quiver.scalar_product_of_simple_roots(
                current_state[i - 1], current_state[i]
            )
            current_state = current_state.act_by_s(i)

        return degree

    @cached_method
    def basis_by_states_and_degrees_tuple(
        self, left_state, right_state, degree, as_tuples=True
    ):
        return tuple(
            self.basis_by_states_and_degrees(
                self, left_state, right_state, degree, as_tuples
            )
        )

    def _get_action_(self, other, op, self_on_left):
        is_left = not self_on_left
        if op == operator.mul:
            if self_on_left is True:
                if self.base() == other:
                    # if self.base().has_coerce_map_from(other):
                    return RightDotAction(other, self, is_left=is_left, op=operator.mul)
                if self.endomorphisms == other:
                    return RightKLRWEndomorphismAction(
                        other, self, is_left=is_left, op=operator.mul
                    )
            else:
                # if self.endomorphisms.has_coerce_map_from(other):
                if self.endomorphisms == other:
                    return LeftKLRWEndomorphismAction(
                        other, self, is_left=is_left, op=operator.mul
                    )

    def braid_set(self):
        return self.KLRWBraid

    def braid(self, state, word):
        return self.braid_set()._element_constructor_(state, word)

    def state_set(self):
        return self.braid_set().KLRWstate_set

    def state(self, iterable):
        return self.state_set()._element_constructor_(iterable)

    @lazy_attribute
    def endomorphisms(self):
        number_of_moving_strands = 0
        for node, dim in self.quiver.dimensions().items():
            if not node.is_framing():
                number_of_moving_strands += dim
        return KLRWEndomorphismAlgebra(self, number_of_moving_strands)

    def clear_cache(self, verbose=True):
        """
        Clears cache from all @cached_method's
        """
        from sage.misc.cachefunc import CachedMethodCaller

        for name, m in self.__dict__.items():
            if isinstance(m, CachedMethodCaller):
                m.clear_cache()
                if verbose:
                    print("Cleared cache of", name)

    # slightly modified coercion defined in CombinatorialFreeModule
    #    def _coerce_map_from_(self, A):
    #        if isinstance(A, KLRWAlgebra):
    #            if self.base().has_coerce_map_from(A.base()):
    #                return True

    #        return super()._coerce_map_from_(A)

    #    def _element_constructor_(self, x):
    #        if isinstance(x.parent(), KLRWAlgebra):
    #            if self.base() is x.parent().base():
    #                return x
    #            elif self.base().has_coerce_map_from(x.parent().base()):
    #                d = {
    #                    basis: self.base()(coef)
    #                    for basis, coef in x._monomial_coefficients.items()
    #                    if not self.base()(coef).is_zero()
    #                }
    #                return self._from_dict(d)
    #
    #        return super()._element_constructor_(x)

    # @cached_method
    def KLRWmonomial(self, state, word=()):
        """
        Returns a monomial corresponding to the data
        If  no word is given, it returns the idempotent corresponding to state
        """
        # checks?
        return self.monomial(
            self.KLRWBraid._element_constructor_(state=state, word=word)
        )

    def idempotent(self, state):
        """
        Returns an idempotent corresponding to the state
        """
        return self.KLRWmonomial(state)

    # @cached_method
    def right_multiply_by_s(self, m, i):
        """
        Multiplies by a braid representing an elementary transposition.
        The transposition is on the right.
        m is a KLRW braid.
        i is the index of an elementary transposition
        Main ingredient of computing the product
        """
        return self.sum(self._right_action_of_s_iter_(m, i))

    def _right_action_of_s_iter_(self, m, i):
        # looking at (i-1)-th and i-th elements in the sequence
        old_right_state = m.right_state()
        left_color, right_color = old_right_state[i - 1], old_right_state[i]

        new_right_state = old_right_state.act_by_s(i)
        intersection_index = m.find_intersection(i - 1, i, right_state=True)
        # braid_after_intersection is the braid is the braid after
        # the intersection of i-th and j-th strands
        # after the existing intersection if they intersect
        # after the new intersection if they don't intersect
        if intersection_index == -1:
            # if don't intersect
            (
                new_intersection_index,
                new_intersection_position,
            ) = m.position_for_new_s_from_right(i)
            # new_intersection_index -= 1
            braid_after_intersection = m[new_intersection_index:]
        else:
            new_intersection_index = intersection_index + 1
            braid_after_intersection = m[intersection_index + 1 :]

        current_state = new_right_state  # braid_after_intersection.right_state()
        left_strand_position_among_same_color = old_right_state.index_among_same_color(
            i - 1
        )
        word_after_intersection = braid_after_intersection.word()
        current_ind = len(word_after_intersection) - 1
        for ind, color in braid_after_intersection.intersections_with_given_strand(
            i - 1
        ):
            if color == left_color:
                d_ij = -self.quiver[left_color, right_color]
                if d_ij > 0:
                    current_state = self.KLRWBraid.find_state_on_other_side(
                        state=current_state,
                        word=word_after_intersection[ind + 1 : current_ind + 1],
                        reverse=True,
                    )
                    current_ind = ind
                    current_ind_in_original_word = ind + new_intersection_index
                    r_i = self.base().variables[left_color].monomial
                    t_ij = self.base().variables[left_color, right_color].monomial
                    # obsolete, but in the else case we can mltiply faster
                    # because there are no dots
                    lower_part = (-r_i * t_ij) * self.KLRWmonomial(
                        state=current_state,
                        word=m.word()[: current_ind_in_original_word - 1],
                    )
                    if d_ij > 1:
                        x_left = (
                            self.base()
                            .variables[
                                left_color, left_strand_position_among_same_color - 1
                            ]
                            .monomial
                        )
                        x_right = (
                            self.base()
                            .variables[
                                left_color, left_strand_position_among_same_color
                            ]
                            .monomial
                        )
                        lower_part = lower_part * sum(
                            (x_left**k) * (x_right ** (d_ij - 1 - k))
                            for k in range(d_ij)
                        )

                    yield lower_part * self.KLRWmonomial(
                        state=new_right_state, word=word_after_intersection[ind + 1 :]
                    )
                # since we crossed another strand with the same color,
                # the position among the same color changes
                left_strand_position_among_same_color -= 1

        current_state = self.KLRWBraid.find_state_on_other_side(
            state=current_state,
            word=word_after_intersection[: current_ind + 1],
            reverse=True,
        )

        if intersection_index == -1:
            index, position = m.position_for_new_s_from_right(i)
            new_word = m.word()[:index] + (position,) + m.word()[index:]
            assert len(new_word) == len(m.word()) + 1
            yield self.KLRWmonomial(state=new_right_state, word=new_word)
        # if do intersect
        else:
            if left_color != right_color:
                d_ij = -self.quiver[left_color, right_color]
                if d_ij >= 0:
                    t_ij = self.base().variables[left_color, right_color].monomial
                    lower_part = self.KLRWmonomial(
                        state=current_state, word=m.word()[:intersection_index]
                    )
                    if d_ij > 0:
                        d_ji = -self.quiver[right_color, left_color]
                        t_ji = self.base().variables[right_color, left_color].monomial
                        x_left = (
                            self.base()
                            .variables[
                                left_color, left_strand_position_among_same_color
                            ]
                            .monomial
                        )
                        right_strand_position_among_same_color = (
                            current_state.index_among_same_color(
                                m.word()[intersection_index] - 1
                            )
                        )
                        x_right = (
                            self.base()
                            .variables[
                                right_color, right_strand_position_among_same_color
                            ]
                            .monomial
                        )
                        lower_part = lower_part * (
                            t_ij * x_left ** (d_ij) + t_ji * x_right ** (d_ji)
                        )
                    else:
                        lower_part = t_ij * lower_part
                    yield lower_part * self.KLRWmonomial(
                        state=new_right_state, word=word_after_intersection
                    )

    @cached_method
    def product_on_basis(self, left, right):
        """
        Computes the product of two KLRW braids
        This method is called by _mul_
        left an right are KLRWbraids
        """
        if left.right_state() != right.left_state():
            if self.warnings:
                print(
                    "states don't match!"
                    + repr(left.right_state())
                    + repr(right.left_state())
                    + "\n"
                    + repr(left)
                    + " * "
                    + repr(right)
                )
            return self.zero()

        if not right.word():
            return self.monomial(left)

        # if the left word is empty
        if not left.word():
            return self.monomial(right)

        # if right has no dots and has non-trivial word
        # one can show that if right.word() is lexmin shortest,
        # then right[1:].word() is also lexmin shortest
        return self.right_multiply_by_s(left, right[0]) * self.monomial(right[1:])

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

    def _repr_(self):
        """
        Single-underscore method for string representation
        """
        return "KLRW algebra"


###############################################################################


@dataclass(slots=True)
class KLRWAlgebraGradedComponent(UniqueRepresentation):
    KLRW_algebra: KLRWAlgebra
    left_state: KLRWstate
    right_state: KLRWstate
    degree: int

    #    def __getattr__(self, name):
    #        if hasattr(self.KLRW_algebra, name):
    #            return getattr(self.KLRW_algebra, name)

    def dimension(self):
        return len(self.basis(as_tuples=False))

    @cached_method
    def basis(self, as_tuples=False):
        """
        Generates a basis in a component
        If as_tuples=True returns tuples (braid, exp)
        representing monomials.
        braid is a KLRW braid.
        exp is a tuple of exponents in the polynomial coefficients.
        """
        if not as_tuples:
            # MappingProxyType makes sure it's immutable
            return MappingProxyType(dict(enumerate(self._basis_iter_())))
        else:
            result = {}
            for i, basis_element in self.basis(as_tuples=False).items():
                assert len(basis_element) == 1
                for braid, monomial in basis_element:
                    assert monomial.is_term()
                    for exp, coeff in monomial.iterator_exp_coeff(as_ETuples=True):
                        assert coeff == self.KLRW_algebra.base().base().one()
                        result[i] = (braid, exp)
            return MappingProxyType(result)

    @lazy_attribute
    def _word_exp_to_index_(self) -> dict:
        """
        Makes a dictionary where for each word and tuple of dot exponents
        we get the index of a basis element representing this combination
        Note that since we fixed left and right state word is the only
        part of the braid data that differs
        """
        result = {}
        for i, (braid, exp) in self.basis(as_tuples=True).items():
            result[braid.word(), exp] = i
        return MappingProxyType(result)

    def _basis_iter_(self):
        for braid in self.KLRW_algebra.KLRWBraid._braids_with_left_state_iter_(
            self.left_state
        ):
            if braid.right_state() == self.right_state:
                for monomial in self.KLRW_algebra.base().dots_of_degree(
                    self.degree - self.KLRW_algebra.braid_degree(braid)
                ):
                    assert monomial.is_term()
                    yield self.KLRW_algebra.term(braid, monomial)

    def _element_from_vector_(self, vector):
        assert len(vector) == len(self.basis())
        assert (
            self.KLRW_algebra.base().base().has_coerce_map_from(vector.parent().base())
        )
        basis = self.basis(as_tuples=False)
        result = self.KLRW_algebra.zero()
        for i, c in enumerate(vector):
            result += c * basis[i]
        return result

    def _vector_from_element_(self, element):
        assert element.parent() is self.KLRW_algebra
        vec = vector(self.KLRW_algebra.base().base(), len(self.basis()))
        for braid, poly in element:
            assert braid.left_state() == self.left_state
            assert braid.right_state() == self.right_state
            for exp, coeff in poly.iterator_exp_coeff(as_ETuples=True):
                index = self._word_exp_to_index_[braid.word(), exp]
                vec[index] += coeff
        return vec

    @cached_method
    def basis_as_matrix_in_graded_component(self, braid, coeff, acting_on_left=True):
        """
        Gives a matrix of how multiplication by a braid with given dots
        acts on this graded component of a Hom between projectives
        """
        element = coeff * self.KLRW_algebra.monomial(braid)

        domain_basis = self.basis(as_tuples=False)

        codomain_degree = self.degree
        codomain_degree += self.KLRW_algebra.braid_degree(braid)
        codomain_degree += coeff.parent().element_degree(
            coeff, check_if_homogeneous=True
        )

        if acting_on_left:
            codomain_left_state = braid.left_state()
            codomain_right_state = self.right_state
        else:
            codomain_left_state = self.left_state
            codomain_right_state = braid.right_state()
        codomain_graded_component = self.KLRW_algebra[
            codomain_left_state:codomain_right_state:codomain_degree
        ]

        mat_transposed = matrix(
            self.KLRW_algebra.base().base(),
            0,
            codomain_graded_component.dimension(),
            sparse=True,
        )

        for _, basis_elem in domain_basis.items():
            if acting_on_left:
                new_element = element * basis_elem
            else:
                new_element = basis_elem * element
            new_row_in_transposed = codomain_graded_component._vector_from_element_(
                new_element
            )
            mat_transposed = mat_transposed.stack(new_row_in_transposed)

        return mat_transposed.transpose()
