from itertools import product
from collections.abc import Iterable
from dataclasses import dataclass

from types import MappingProxyType
import operator

from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.categories.finite_dimensional_algebras_with_basis import (
    FiniteDimensionalAlgebrasWithBasis,
)
from sage.categories.action import Action
from sage.rings.polynomial.polydict import ETuple
from sage.misc.lazy_attribute import lazy_attribute
from sage.misc.cachefunc import cached_method, weak_cached_function
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector

from .klrw_state import KLRWstate
from .klrw_braid import KLRWbraid, KLRWbraid_set
from .framed_dynkin import FramedDynkinDiagram_with_dimensions
from .dot_algebra import KLRWDotsAlgebra
from .gradings import (
    QuiverGradingGroupElement,
    QuiverGradingGroup,
)
from .bimodule_monoid import LeftFreeBimoduleMonoid, RightActionOnBimodule
from .opposite_algebra import OppositeAlgebra


class KLRWElement(IndexedFreeModuleElement):
    def degree(self, check_if_homogeneous=False):
        # zero elements return None as degree
        degree = None
        for braid, coeff in self:
            if not coeff.is_zero():
                braid_degree = self.parent().braid_degree(braid)
                coeff_degree = (
                    self.parent()
                    .base()
                    .element_degree(
                        coeff,
                        grading_group=self.parent().grading_group,
                        check_if_homogeneous=check_if_homogeneous,
                    )
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

    def is_geometric(self):
        """
        Return True if self has at least one non-zero constant term.
        """
        for _, coeff in self:
            if not coeff.constant_coefficient().is_zero():
                return True

        return False

    def geometric_part(self):
        """
        Return the geometric part of the element.

        The geometric part contains only constant dot coefficients.
        """
        dot_algebra = self.parent().base()
        return self.parent().sum_of_terms(
            (
                braid,
                dot_algebra.geometric_part(coeff),
            )
            for braid, coeff in self
        )

    def _matmul_(self, other):
        """
        Used *only* in the special case when both self and other
        have the same parent.
        Because of a technical details in the coercion model
        this method is called instead of action.
        """
        return self.parent()._self_action.act(self, other)

    # TODO:rewrite
    # def check(self):

    # TODO: add checks that all braids are correct


class RightDotAction(RightActionOnBimodule):
    # @cached_method
    # def _act_(self, p, x: IndexedFreeModuleElement) -> IndexedFreeModuleElement:
    #     p = self.codomain().base()(p)
    #     return self.codomain().linear_combination(
    #         (self._act_on_bases_(exp_tuple, braid), coeff * left_poly)
    #         for exp_tuple, coeff in p.iterator_exp_coeff()
    #         for braid, left_poly in x
    #     )

    def _act_on_base_in_bimodule_iter_(self, p, x_index):
        # `x_index` is a braid in this setting.
        p = self.codomain().base()(p)
        for exp_tuple, coeff in p.iterator_exp_coeff():
            for new_index, new_coeff in self._act_on_bases_(exp_tuple, x_index):
                yield (new_index, coeff * new_coeff)

    @cached_method
    def _act_on_bases_(self, p_exp_tuple: ETuple, braid: KLRWbraid):
        return self.codomain().sum(self._act_on_bases_iter_(p_exp_tuple, braid))

    def _act_on_bases_iter_(self, p_exp_tuple: ETuple, braid: KLRWbraid):
        # idempotents commute with dots
        if len(braid.word()) == 0:
            yield self.codomain().term(
                braid, self.codomain().base().monomial(*p_exp_tuple)
            )
        else:
            last_letter = braid.word()[-1]
            c1, c2 = braid.intersection_colors(-1)
            if c1 == c2:
                index_of_dots_on_left_strand = (
                    self.codomain()
                    .base()
                    .dot_variable(
                        c1,
                        braid.right_state().index_among_same_color(last_letter - 1),
                    )
                    .position
                )
                index_of_special_coefficient = (
                    self.codomain().base().vertex_variable(c1).position
                )
                # terms where crossing stays
                coeff_of_crossing = self._coeff_of_crossing_(
                    p_exp_tuple, index_of_dots_on_left_strand
                )
                for exp_tuple, coeff in coeff_of_crossing.iterator_exp_coeff():
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
                    for exp_tuple, coeff in poly.iterator_exp_coeff():
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
        return self.codomain().base().monomial(*new_exp)

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
                coeff = (
                    self.codomain().scalars()(delta)
                    * self.codomain().base().default_edge_parameter
                )
            else:
                coeff = self.codomain().scalars()(delta)
                new_exp[j] += 1

            while new_exp[i] != end_degree_at_i:
                # we do tuple() to make a copy
                # and plug it in later into :meth:G.monomial
                yield coeff * self.codomain().base().monomial(*tuple(new_exp))
                new_exp[i] += -delta
                new_exp[i + 1] += delta


class KLRWAlgebra(LeftFreeBimoduleMonoid):
    @weak_cached_function(cache=128)  # automatically a staticmethod
    #@staticmethod
    def __classcall_private__(
        cls,
        base_R,
        quiver_data: FramedDynkinDiagram_with_dimensions,
        vertex_scaling=False,
        edge_scaling=False,
        dot_scaling=False,
        invertible_parameters=False,
        dot_algebra_order="degrevlex",
        warnings=False,
        **kwrds,
    ):
        """
        Returns a new instance.

        We use extra caching co
        """
        return super().__classcall__(
            cls,
            base_R=base_R,
            quiver_data=quiver_data.immutable_copy(),
            vertex_scaling=vertex_scaling,
            edge_scaling=edge_scaling,
            dot_scaling=dot_scaling,
            invertible_parameters=invertible_parameters,
            dot_algebra_order=dot_algebra_order,
            warnings=warnings,
            **kwrds,
        )

    def __init__(
        self,
        base_R,
        quiver_data: FramedDynkinDiagram_with_dimensions,
        vertex_scaling=False,
        edge_scaling=False,
        dot_scaling=False,
        invertible_parameters=False,
        dot_algebra_order="degrevlex",
        warnings=False,
        **kwrds,
    ):
        """
        Makes a KLRW algebra for a quiver.
        base_R is the base ring.
        if warnings, then extra warnings can be printed

        Warning: default parameters in the dot algebra
        may work incorrectly with extra gradings
        """
        self.warnings = warnings
        self.quiver = quiver_data.quiver
        self.KLRWBraid = KLRWbraid_set(quiver_data, state_on_right=True)

        self.grading_group = QuiverGradingGroup(
            self.quiver,
            vertex_scaling=vertex_scaling,
            edge_scaling=edge_scaling,
            dot_scaling=dot_scaling,
        )

        dots_algebra = KLRWDotsAlgebra(
            base_R,
            quiver_data,
            invertible_parameters=invertible_parameters,
            order=dot_algebra_order,
            **kwrds,
        )
        category = FiniteDimensionalAlgebrasWithBasis(dots_algebra)
        super().__init__(R=dots_algebra, element_class=KLRWElement, category=category)

    def __getitem__(self, key: slice):
        """
        For an instance KLRW of KLRW algebra
        KLRW[left_state:right_state:degree] is
        the graded finite-dimensional component
        """
        if isinstance(key, slice):
            return self.graded_component(
                key.start, key.stop, self.grading_group(key.step)
            )

    # cache to keep hard references for graded components
    @cached_method
    def graded_component(
        self,
        left_state: KLRWstate,
        right_state: KLRWstate,
        degree: QuiverGradingGroupElement,
    ):
        return KLRWAlgebraGradedComponent(self, left_state, right_state, degree)

    @lazy_attribute
    def ideal_of_symmetric_dots(self):
        return list(self.base().symmetric_dots_gens()) * self.base()

    def modulo_symmetric_dots(self, element):
        return element.reduce(self.ideal_of_symmetric_dots)

    def one(self):
        raise ValueError()
        if self.warnings:
            print(
                r"The identity of KLRW is used' often this is done deep in Sage's code."
                + "\n"
                + r"It's faster to replace this by a function that does not use it."
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

    def from_base_ring(self, r):
        """
        Return the canonical embedding of ``r`` into ``self``.
        """
        if r.is_zero():
            return self.zero()
        return self.one()._lmul_(r)

    """
    def linear_combination(self, iter_of_elements_coeff):
        from sage.data_structures.blas_dict import iaxpy
        result = {}

        for element, coeff in iter_of_elements_coeff:
            monomial_coefficients = element._monomial_coefficients
            if not coeff: # We multiply by 0, so nothing to do
                continue
            if not result:
                if coeff == 1:
                    result = monomial_coefficients.copy()
                    continue
            iaxpy(coeff, monomial_coefficients, result, remove_zeros=False)

        # generator instead of a list & removal
        if any(val.is_zero() for val in result.values()):
            # self._counter_()
            result = dict(
                item for item in result.items() if not item[1].is_zero()
            )

        #filtered_result = dict(
        #    filter(lambda item: not item[0].is_zero(), result.items())
        #)

        return self._from_dict(
            result,
            coerce=True,
            remove_zeros=False,
        )
    """

    # def _counter_(self):
    #    pass

    def scalars(self):
        return self.base().base()

    def gens(self):
        # dots and other coefficients
        for state in self.KLRWBraid.KLRWstate_set:
            for scalar in self.base().gens():
                yield self.term(self.KLRWBraid._element_constructor_(state), scalar)
        # simple crossings
        yield from self.gens_over_dots()

    def gens_over_dots(self):
        # simple crossings
        for state in self.KLRWBraid.KLRWstate_set:
            for i in range(1, len(state)):
                if not state[i - 1].is_framing() and not state[i].is_framing():
                    yield self.term(
                        self.KLRWBraid._element_constructor_(state, (i,)),
                        self.base().one(),
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

    @cached_method
    def braid_degree(self, braid):
        degree = 0
        current_state = braid.right_state()

        for i in reversed(braid.word()):
            degree += self.grading_group.crossing_grading(
                current_state[i - 1], current_state[i]
            )
            current_state = current_state.act_by_s(i)

        return degree

    def element_max_number_of_dots(self, element):
        n_dots = 0
        for braid, coeff in element:
            term_dots = coeff.parent().element_max_number_of_dots(coeff)

            if term_dots > n_dots:
                n_dots = term_dots

        return n_dots

    def scale_dots_in_element(self, element, multipliers):
        dots_algebra = self.base()
        return self.sum_of_terms(
            (
                braid,
                dots_algebra.scale_dots_in_element(coeff, multipliers),
            )
            for braid, coeff in element
        )

    @cached_method
    def basis_by_states_and_degrees_tuple(
        self, left_state, right_state, degree, as_tuples=True
    ):
        return tuple(
            self.basis_by_states_and_degrees(
                self, left_state, right_state, degree, as_tuples
            )
        )

    def _right_base_action_(self, other):
        return RightDotAction(other, self, is_left=False, op=operator.mul)

    def _get_action_(self, other, op, self_on_left):
        # if op == operator.mul:
        #     if self_on_left is True:
        #         if self.base().has_coerce_map_from(other):
        #             return RightDotAction(
        #                 other, self, is_left=not self_on_left, op=operator.mul
        #             )
        if op == operator.matmul:
            if self_on_left:
                if isinstance(other, KLRWAlgebra):
                    return TensorMultiplication(left_parent=self, right_parent=other)
        return super()._get_action_(other, op, self_on_left)

    def braid_set(self):
        return self.KLRWBraid

    def braid(self, state, word):
        return self.braid_set()._element_constructor_(state, word)

    def state_set(self):
        return self.braid_set().KLRWstate_set

    def state(self, iterable):
        return self.state_set()._element_constructor_(iterable)

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

    def _coerce_map_from_(self, other):
        if isinstance(other, KLRWAlgebra):
            can_coerce = self.braid_set().has_coerce_map_from(other.braid_set())
            can_coerce &= self.base().has_coerce_map_from(other.base())
            if can_coerce:
                return lambda parent, x: parent._from_dict(
                    {
                        parent.braid_set().coerce(braid): parent.base().coerce(coeff)
                        for braid, coeff in x
                    }
                )
            return

        return super()._coerce_map_from_(other)

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
                    r_i = self.base().vertex_variable(left_color).monomial
                    t_ij = self.base().edge_variable(left_color, right_color).monomial
                    # obsolete, but in the else case we can mltiply faster
                    # because there are no dots
                    lower_part = (-r_i * t_ij) * self.KLRWmonomial(
                        state=current_state,
                        word=m.word()[: current_ind_in_original_word - 1],
                    )
                    if d_ij > 1:
                        x_left = (
                            self.base()
                            .dot_variable(
                                left_color,
                                left_strand_position_among_same_color - 1,
                            )
                            .monomial
                        )
                        x_right = (
                            self.base()
                            .dot_variable(
                                left_color,
                                left_strand_position_among_same_color,
                            )
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
                    t_ij = self.base().edge_variable(left_color, right_color).monomial
                    lower_part = self.KLRWmonomial(
                        state=current_state, word=m.word()[:intersection_index]
                    )
                    if d_ij > 0:
                        d_ji = -self.quiver[right_color, left_color]
                        t_ji = (
                            self.base().edge_variable(right_color, left_color).monomial
                        )
                        x_left = (
                            self.base()
                            .dot_variable(
                                left_color,
                                left_strand_position_among_same_color,
                            )
                            .monomial
                        )
                        right_strand_position_among_same_color = (
                            current_state.index_among_same_color(
                                m.word()[intersection_index] - 1
                            )
                        )
                        x_right = (
                            self.base()
                            .dot_variable(
                                right_color,
                                right_strand_position_among_same_color,
                            )
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
                raise ValueError()
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

    def element_from_braid(
        self,
        word: Iterable,
        left_state=None,
        right_state=None,
        coeff=None,
    ):
        """
        Creates an element form any braid data.

        I.e. the braid does not have to be of special form
        [minimal, lexmin]
        Works slowlier than KLRWmonomial
        that requires the braid to be minimal lexmin.
        """
        assert (
            left_state is not None or right_state is not None
        ), "Left or right state must be given"

        if right_state is not None:
            state = right_state
            result = self.idempotent(state)
            for i in reversed(word):
                crossing_element = self.KLRWmonomial(
                    state=state,
                    word=(i,),
                )
                result = crossing_element * result
                state = state.act_by_s(i)

            if left_state is not None:
                assert state == left_state, "Left and right states are incompatible."

        else:
            # the case where only the left side was given
            state = left_state
            result = self.idempotent(state)
            for i in word:
                state = state.act_by_s(i)
                crossing_element = self.KLRWmonomial(
                    state=state,
                    word=(i,),
                )
                result = result * crossing_element

        if coeff is not None:
            result = coeff * result
        return result

    @lazy_attribute
    def opposite(self):
        return OppositeAlgebra(self)

    def klrw_options(self):
        """
        Return keywords needed for creating this
        (or isomorphic, if called in subclasses)
        KLRW algebra.
        """
        assert not self._reduction[1]
        return self._reduction[2].copy()

    def _replace_(self, **replacements):
        """
        Make a similar parent with several adjustments.

        Compare to _replace of named tuples.
        Warning: this makes an instance of KLRWAlgebra,
        not its subclasses (e.g. `TensorProductOfKLRWAlgebras`)
        """
        from sage.structure.unique_representation import unreduce

        kwrds = self.klrw_options()
        new_kwrds = kwrds | replacements
        return unreduce(KLRWAlgebra, (), new_kwrds)

    @staticmethod
    def tensor_product(*terms):
        return TensorProductOfKLRWAlgebras(*terms)

    def __matmul__(self, other):
        return self.tensor_product(self, other)

    @lazy_attribute
    def _self_action(self):
        return TensorMultiplication(left_parent=self, right_parent=self)


###############################################################################


# Instead of inheritance from `UniqueRepresentation`
# We cache in corresponding KLRW_Algebra.
# This is because `UniqueRepresentation` uses weak cache
# and we have many cases when we don't keep
# a strong reference to graded components.
@dataclass(slots=True)
class KLRWAlgebraGradedComponent:
    KLRW_algebra: KLRWAlgebra
    left_state: KLRWstate
    right_state: KLRWstate
    degree: QuiverGradingGroupElement

    #    def __getattr__(self, name):
    #        if hasattr(self.KLRW_algebra, name):
    #            return getattr(self.KLRW_algebra, name)

    def dimension(self):
        return len(self.basis(as_tuples=False))

    @cached_method
    def basis(
        self,
        relevant_coeff_degree=None,
        relevant_parameter_part=None,
        as_tuples=False,
    ):
        """
        Generates a basis in a component.

        If as_tuples=True returns tuples
        `(braid, exp)`
        representing monomials.
        `braid` is a KLRW braid.
        `exp` is a tuple of exponents in the polynomial coefficients.

        If `relevant_coeff_degree` is given, sorts only the entries
        where `exp` has the given degree.
        If `relevant_parameter_part` is given, sorts only the entries
        where `exp` has the given parameter (i.e. non-dot) part.
        """
        if as_tuples:
            return tuple(
                self._basis_iter_as_tuples_(
                    relevant_coeff_degree,
                    relevant_parameter_part,
                )
            )
        else:
            result = []
            for braid, exp in self._basis_iter_as_tuples_(
                relevant_coeff_degree, relevant_parameter_part
            ):
                coeff = self.KLRW_algebra.base().monomial(*exp)
                result.append(self.KLRW_algebra.term(braid, coeff))
            return tuple(result)

    @cached_method
    def _word_exp_to_index_(self) -> MappingProxyType:
        """
        Makes a dictionary where for each word and tuple of dot exponents
        we get the index of a basis element representing this combination
        Note that since we fixed left and right state word is the only
        part of the braid data that differs
        """
        result = {}
        for i, (braid, exp) in enumerate(self.basis(as_tuples=True)):
            result[braid.word(), exp] = i
        return MappingProxyType(result)

    def _basis_iter_as_tuples_(
        self,
        relevant_coeff_degree=None,
        relevant_parameter_part=None,
    ):
        """
        Iterates over basis monomials.

        Yields pairs
        `(braid, exp)`
        where `braid` is the braid part, and `exp` is the ETuple
        representing a monomial in the dots_algebra.

        If `relevant_coeff_degree` is given, sorts only the entries
        where `exp` has the given degree.
        If `relevant_parameter_part` is given, sorts only the entries
        where `exp` has the given parameter (i.e. non-dot) part.
        """
        dot_algebra = self.KLRW_algebra.base()
        for braid in self.KLRW_algebra.KLRWBraid.braids_by_states(
            self.left_state, self.right_state
        ):
            dot_degree = self.degree - self.KLRW_algebra.braid_degree(braid)
            if relevant_coeff_degree is not None:
                if dot_degree != relevant_coeff_degree:
                    continue
            if relevant_parameter_part is None:
                for exp in dot_algebra.exps_by_degree(
                    self.degree - self.KLRW_algebra.braid_degree(braid),
                ):
                    yield (braid, exp)
            else:
                for exp in dot_algebra.exps_by_degree_and_parameter_part(
                    self.degree - self.KLRW_algebra.braid_degree(braid),
                    relevant_parameter_part,
                ):
                    yield (braid, exp)

    def _element_from_vector_(self, vector):
        assert len(vector) == len(self.basis())
        assert self.KLRW_algebra.scalars().has_coerce_map_from(vector.parent().base())
        basis = self.basis(as_tuples=False)
        result = self.KLRW_algebra.zero()
        for i, c in enumerate(vector):
            result += c * basis[i]
        return result

    def _vector_from_element_(self, element):
        assert element.parent() is self.KLRW_algebra
        vec = vector(self.KLRW_algebra.scalars(), len(self.basis()))
        for braid, poly in element:
            assert braid.left_state() == self.left_state, (
                repr(braid) + " " + repr(self.left_state) + " " + repr(self.right_state)
            )
            assert braid.right_state() == self.right_state, (
                repr(braid) + " " + repr(self.left_state) + " " + repr(self.right_state)
            )
            for exp, coeff in poly.iterator_exp_coeff():
                index = self._word_exp_to_index_()[braid.word(), exp]
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
        codomain_degree += element.degree(check_if_homogeneous=True)

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
            self.KLRW_algebra.scalars(),
            0,
            codomain_graded_component.dimension(),
            sparse=True,
        )

        for basis_elem in domain_basis:
            if acting_on_left:
                new_element = element * basis_elem
            else:
                new_element = basis_elem * element
            new_row_in_transposed = codomain_graded_component._vector_from_element_(
                new_element
            )
            mat_transposed = mat_transposed.stack(new_row_in_transposed)

        return mat_transposed.transpose()


###############################################################################


class TensorProductOfKLRWAlgebras(KLRWAlgebra):
    """
    Tensor product over the parameter algebra.

    Warning:
    So far it's implemented as the ambient KLRW algebra.
    Can override methods like `_element_constructor_`
    and `gens` to get correct behaviour.
    """

    def __init__(self, *klrw_algebras):
        from klrw.misc import get_from_all_and_assert_equality
        from klrw.dot_algebra import TensorProductOfDotAlgebras

        assert klrw_algebras, "Need at least one KLRW algebra"
        assert all(isinstance(alg, KLRWAlgebra) for alg in klrw_algebras)
        self._parts = klrw_algebras

        self.warnings = get_from_all_and_assert_equality(
            lambda x: x.warnings,
            self._parts,
        )

        dots_algebra = TensorProductOfDotAlgebras(*(alg.base() for alg in self._parts))

        quiver_data = dots_algebra.quiver_data
        self.quiver = quiver_data.quiver
        self.KLRWBraid = KLRWbraid_set(quiver_data, state_on_right=True)

        self.grading_group = get_from_all_and_assert_equality(
            lambda x: x.grading_group,
            self._parts,
        )

        category = FiniteDimensionalAlgebrasWithBasis(dots_algebra)
        LeftFreeBimoduleMonoid.__init__(
            self, R=dots_algebra, element_class=KLRWElement, category=category
        )

    def klrw_options(self):
        """
        Return keywords needed for creating this
        (or isomorphic, if called in subclasses)
        KLRW algebra.
        """
        options = {
            "base_R": self.base().base_ring(),
            "quiver_data": self.base().quiver_data,
            "vertex_scaling": self.grading_group.vertex_scaling,
            "edge_scaling": self.grading_group.edge_scaling,
            "dot_scaling": self.grading_group.dot_scaling,
            "invertible_parameters": self.base().invertible_parameters,
            "dot_algebra_order": self.base().term_order(),
            "warnings": self.warnings,
        }
        options |= self.base().prefixes

        return options

    @cached_method
    def ambient_algebra(self):
        """
        Return ambient algebra.
        """

        return self._replace_()

    def _tensor_product_of_braids(self, *braids):
        state = []
        word = []
        for br in braids:
            word += [i + len(state) for i in br.word()]
            state += br.state().as_tuple()
        braid = self.braid(state=self.state(state), word=tuple(word))
        return braid

    def tensor_product_of_elements(self, *elements):
        result = self.zero()
        it = product(*(iter(elem) for elem in elements))
        for (*braid_coeff_pairs,) in it:
            braids, coeffs = zip(*braid_coeff_pairs)
            product_coeff = self.base().one()
            for i, coeff in enumerate(coeffs):
                product_coeff *= self.base().embedding(i)(coeff)
            product_braid = self._tensor_product_of_braids(*braids)
            result += self.term(product_braid, product_coeff)

        return result


class TensorMultiplication(Action):
    """
    Gives multiplication of two KLRW elements.

    The solution is similar to matrix multiplication is Sage.
    [see MatrixMatrixAction in MatrixSpace]
    Coercion model in Sage requires both elements in
    the usual product to have the same parent, but in
    actions this is no longer the case. So we use the class Action.
    """

    def __init__(
        self,
        left_parent: KLRWAlgebra,
        right_parent: KLRWAlgebra,
    ):
        Action.__init__(
            self, G=left_parent, S=right_parent, is_left=True, op=operator.matmul
        )
        self._left_domain = self.actor()
        self._right_domain = self.domain()
        self._codomain = self._left_domain @ self._right_domain

    def _act_(self, left, right):
        return self._codomain.tensor_product_of_elements(left, right)
