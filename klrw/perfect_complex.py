from typing import Iterable, Any
from collections import defaultdict
from dataclasses import dataclass, InitVar
from types import MappingProxyType
from itertools import product

import operator

from sage.structure.parent import Parent

from sage.matrix.matrix0 import Matrix
from sage.matrix.constructor import matrix

from sage.modules.free_module import FreeModule
from sage.structure.element import ModuleElement, Element

from sage.rings.integer_ring import ZZ
from sage.rings.integer import Integer
from sage.rings.ring import Ring

from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute, lazy_class_attribute
from sage.categories.action import Action
from sage.modules.module import Module
from sage.data_structures.blas_dict import add, negate, axpy
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.element import get_coercion_model

from .klrw_state import KLRWstate
from .klrw_algebra import KLRWAlgebra
from .gradings import (
    QuiverGradingGroup,
    QuiverGradingGroupElement,
    HomologicalGradingGroupElement,
    HomologicalGradingGroup,
    ExtendedQuiverGradingGroupElement,
    ExtendedQuiverGradingGroup,
)
from .opposite_algebra import FreeRankOneModule_Endset


class GradedFreeModule(UniqueRepresentation):
    @lazy_class_attribute
    def OriginalClass(cls):
        return cls

    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedGradedFreeModule

    @lazy_class_attribute
    def HomsetClass(cls):
        return GradedFreeModule_Homset

    @staticmethod
    def __classcall__(
        cls,
        ring: Ring,
        component_ranks: dict[Element, int],
        grading_group=ZZ,
        **kwargs,
    ):
        """
        Making a Graded free module from a dictionary {grading: rank}

        Deleting all zero components.
        Convering gradings into elements of `grading_group`
        Making `component_ranks` a frozenset to make it hashable.
        """
        if not isinstance(component_ranks, frozenset):
            component_ranks = {
                grading_group(deg): dim for deg, dim in component_ranks.items() if dim
            }
            component_ranks = frozenset(component_ranks.items())

        return super().__classcall__(
            cls,
            ring=ring,
            component_ranks=component_ranks,
            grading_group=grading_group,
            **kwargs,
        )

    def __init__(
        self,
        ring: Ring,
        component_ranks: frozenset[tuple[Element, int]],
        grading_group,
    ):
        self._ring = ring

        # Setting _component_ranks to be MappingProxyType to avoid modification;
        self._component_ranks = MappingProxyType(
            {grading: dim for grading, dim in component_ranks}
        )
        self._grading_group = grading_group

    def gradings(self):
        return self._component_ranks.keys()

    def component_rank(self, grading):
        grading = self._grading_group(grading)
        if grading in self._component_ranks:
            return self._component_ranks[grading]
        else:
            return 0

    def component_rank_iter(self):
        return self._component_ranks.items()

    def hom_grading_group(self):
        return self._grading_group

    def shift_group(self):
        return self._grading_group

    def ring(self):
        return self._ring

    def __getitem__(self, key):
        shift = self._grading_group(key)

        if shift:
            return self.ShiftedClass(
                self,
                shift=shift,
            )
        else:
            return self

    def __repr__(self):
        from pprint import pformat

        result = "A graded free module with dimensions\n"
        result += pformat(dict(self._component_ranks))

        return result

    def is_instance_of_sameclass_or_shiftedclass(self, other):
        return isinstance(other, self.__class__ | self.ShiftedClass)

    def hom_set(self, other):
        if self.is_instance_of_sameclass_or_shiftedclass(other):
            return self.HomsetClass(domain=self, codomain=other)
        else:
            raise TypeError(
                "Can't define homomorphisms from\n{}\nto\n{}\n".format(self, other)
            )

    def hom(self, other, morphism_data, **kwargs):
        return self.hom_set(other)._element_constructor_(morphism_data, **kwargs)

    def _replace_(self, **replacements):
        """
        Make a similar parent with several adjustments.

        Compare to _replace of named tuples.
        """
        from sage.structure.unique_representation import unreduce

        cls, args, kwrds = self._reduction
        new_kwrds = kwrds | replacements
        return unreduce(cls, args, new_kwrds)

    def base_change(self, other: Ring):
        return self._replace_(
            ring=other,
        )


# Do we need UniqueRepresentation?
# We don't cache anything associated to these,
# so we don't need to do it;
# it might be more useful to have slots.
@dataclass(frozen=True, eq=True, repr=False, slots=True)
class ShiftedObject:
    """
    Objects of this class are supposed to be created by [shift]
    operator of other modules, not by calling init directly.
    """

    original: Any
    shift: Element

    @lazy_class_attribute
    def ShiftedClass(cls):
        return cls

    def shift_object(self, shift):
        """
        Construct a shifted object.

        Parent of `shift` must have a coercion
        to the parent of `self.shift`.
        """
        new_shift = self.shift + shift

        if new_shift:
            return self.__class__(
                self.original,
                shift=new_shift,
            )
        else:
            return self.original

    def __repr__(self):
        result = repr(self.original)
        result += "\n Shifted by {}".format(self.shift)

        return result

    @staticmethod
    def find_shift(one, other):
        """
        Find shift between two objects.

        Return the shift one needs to apply to `one` to
        get the `other`.
        It works if one or both are created by shifting
        equal objects.
        Warning: `UniqueRepresentation` forces objects to
        be equal only if they are created with identical data.

        Return `None` objects are not immediately shifts of each other.
        """
        one_is_shifted = isinstance(one, ShiftedObject)
        other_is_shifted = isinstance(other, ShiftedObject)
        if one_is_shifted:
            one_object = one.original
            one_shift = one.shift
        else:
            one_object = one
            one_shift = 0
        if other_is_shifted:
            other_object = other.original
            other_shift = other.shift
        else:
            other_object = other
            other_shift = 0

        if one_object == other_object:
            return other_shift - one_shift

        return None


class ShiftedGradedFreeModule(ShiftedObject):
    """
    A graded module with shifted gradings.
    """

    @lazy_class_attribute
    def OriginalClass(cls):
        return GradedFreeModule

    def __post_init__(self):
        assert isinstance(self.original, GradedFreeModule)

    def gradings(self):
        return frozenset(grading - self.shift for grading in self.original.gradings())

    def component_rank(self, grading):
        return self.original.component_rank(grading + self.hom_shift())

    def component_rank_iter(self):
        for grading, dim in self.original.component_rank_iter():
            yield (grading - self.shift, dim)

    def hom_grading_group(self):
        return self.original.hom_grading_group()

    def hom_shift(self):
        return self.hom_grading_group()(self.shift)

    def shift_group(self):
        return self.original.shift_group()

    def ring(self):
        return self.original.ring()

    def __getitem__(self, key):
        extra_shift = self.shift_group()(key)
        return self.shift_object(extra_shift)

    def is_instance_of_sameclass_or_shiftedclass(self, other):
        return self.original.is_instance_of_sameclass_or_shiftedclass(other)

    def hom_set(self, other, use_shift_isomorphism=False):
        if use_shift_isomorphism:
            return self.original.hom_set(other[-self.shift])
        else:
            return self.original.HomsetClass(self, other)

    def hom(self, other, morphism_data):
        raise NotImplementedError()

    def base_change(self, other: Ring):
        new_original = self.original.base_change(other)
        new_shift_group = new_original.shift_group()
        new_shift = new_shift_group(self.shift)
        return self.__class__(
            new_original,
            new_shift,
        )


class GradedFreeModule_Homomorphism(ModuleElement):
    __slots__ = ("_map_components",)

    def __init__(
        self,
        M: Parent,
        x: Iterable | dict,
        check=True,
        copy=True,
    ):
        ModuleElement.__init__(self, M)

        if not copy:
            if isinstance(x, GradedFreeModule_Homomorphism):
                self._map_components = x._map_components
            elif isinstance(x, dict):
                self._map_components = x
            else:
                raise ValueError(
                    "Don't know how to make "
                    + repr(x)
                    + " into a homomorphism without copying."
                )

        else:
            self._map_components = {}
            end_algebra = self.parent().end_algebra
            if isinstance(x, dict):
                x = x.items()
            for deg, map in x:
                deg = self.parent().hom_grading_group()(deg)
                if isinstance(map, Matrix):
                    assert map.ncols() == M.domain.component_rank(deg)
                    assert map.nrows() == M.codomain.component_rank(deg)
                    map = map.change_ring(end_algebra)
                    self._map_components[deg] = map
                    self._map_components[deg].set_immutable()
                else:
                    self._map_components[deg] = matrix(
                        end_algebra,
                        ncols=M.domain.component_rank(deg),
                        nrows=M.codomain.component_rank(deg),
                        entries=map,
                        sparse=True,
                        immutable=True,
                    )

        # check conditions like commuting with differential
        # or having correct grading
        if check:
            print("++++check")
            self.parent()._element_check_(self)
        else:
            print("----check")

    def _add_(self, other):
        return self.__class__(
            self.parent(),
            add(
                self._map_components,
                other._map_components,
            ),
            check=False,
            copy=False,
        )

    def _neg_(self):
        return self.__class__(
            self.parent(),
            negate(self._map_components),
            check=False,
            copy=False,
        )

    def _sub_(self, other):
        return self.__class__(
            self.parent(),
            axpy(
                -1,
                other._map_components,
                self._map_components,
            ),
            check=False,
            copy=False,
        )

    def __iter__(self):
        yield from self._map_components.items()

    def __call__(self, grading, sign=False):
        """
        Return a component of the homomorphism.
        """
        if grading in self._map_components:
            if not sign:
                return self._map_components[grading]
            else:
                return -self._map_components[grading]
        return None

    def __getitem__(self, shift):
        hom_shift = self.parent().hom_grading_group()(shift)
        shifted_components = {
            grading - hom_shift: mat for grading, mat in self._map_components.items()
        }
        return self.__class__(
            self.parent().shift(shift),
            shifted_components,
            check=False,
            copy=False,
        )

    def _mul_(self, other):
        """
        Used *only* in the special case when both self and other
        have the same parent.
        Because of a technical details in the coercion model
        this method is called instead of action.
        """
        return self.parent()._self_action.act(self, other)

    def _rmul_(self, left: Element):
        """
        Reversed scalar multiplication for module elements with the
        module element on the right and the scalar on the left.

        Warning: one has to make sure that multiplication by `left`
        in the base ring does not ruin conditions checked in
        `self.parent()._element_check_`.
        E.g. gradings, commuting with differentials.
        """
        # we write this instead of using sage.data_structures.blas_dict
        # bacause matrices handle multiplication by a scalar by
        # coercing it to an element of the base algebra
        # which is not optimal for KLRW.
        result_dict = {}
        for deg, map in self._map_components.items():
            modified_map_dict = {
                position: left * elem for position, elem in map.dict(copy=False).items()
            }
            modified_map = matrix(
                self.parent().end_algebra,
                nrows=map.nrows(),
                ncols=map.ncols(),
                entries=modified_map_dict,
                sparse=True,
                immutable=True,
            )
            if modified_map:
                result_dict[deg] = modified_map
        return self.__class__(
            self.parent(),
            result_dict,
            check=False,
            copy=False,
        )

    def _lmul_(self, right: Element):
        """
        Scalar multiplication for module elements with the module
        element on the left and the scalar on the right.

        Warning: one has to make sure that multiplication by `right`
        in the base ring does not ruin conditions checked in
        `self.parent()._element_check_`.
        E.g. gradings, commuting with differentials.
        """
        # we write this instead of using sage.data_structures.blas_dict
        # bacause matrices handle multiplication by a scalar by
        # coercing it to an element of the base algebra
        # which is not optimal for KLRW.
        result_dict = {}
        for deg, map in self._map_components.items():
            modified_map_dict = {
                position: elem * right
                for position, elem in map.dict(copy=False).items()
            }
            modified_map = matrix(
                self.parent().end_algebra,
                nrows=map.nrows(),
                ncols=map.ncols(),
                entries=modified_map_dict,
                sparse=True,
                immutable=True,
            )
            if modified_map:
                result_dict[deg] = modified_map
        return self.__class__(
            self.parent(),
            result_dict,
            check=False,
            copy=False,
        )

    def _repr_(self):
        if not self._map_components:
            return "0"
        result = ""
        for grading in sorted(self._map_components.keys()):
            result += ">" + repr(grading) + ":\n"
            result += repr(self._map_components[grading]) + "\n"

        return result

    def __hash__(self):
        return hash(frozenset(self._map_components.items()))

    def __bool__(self):
        return bool(self._map_components)

    def dict(self):
        return self._map_components

    def items(self):
        return self._map_components.items()

    def domain(self):
        return self.parent().domain

    def codomain(self):
        return self.parent().codomain

    def support(self):
        return self._map_components.keys()

    def differential(self):
        return self.parent().apply_differential(self)


class HomHomMultiplication(Action):
    """
    Gives multiplication of two homs of graded projectives.

    The solution is similar to matrix multiplication is Sage.
    [see MatrixMatrixAction in MatrixSpace]
    Coercion model in Sage requires both elements in
    the usual product to have the same parent, but in
    actions this is no longer the case. So we use the class Action.
    """

    def __init__(
        self,
        left_parent: GradedFreeModule,
        right_parent: GradedFreeModule,
    ):
        Action.__init__(
            self, G=left_parent, S=right_parent, is_left=True, op=operator.mul
        )
        self._left_domain = self.actor()
        self._right_domain = self.domain()
        self._codomain, self._shift = self._find_codomain_and_shift()
        self._hom_shift = self._right_domain.hom_grading_group()(self._shift)

    def _find_codomain_and_shift(self):
        shift = ShiftedObject.find_shift(
            self._left_domain.domain,
            self._right_domain.codomain,
        )
        if shift is None:
            raise TypeError("Incomposable homs")

        # finding a common superclass
        # it will be a subclass of GradedFreeModule
        # so the checks are automatic
        common_superclasses = (
            cls
            for cls in self._left_domain._class_without_category_().__mro__
            if cls in self._left_domain._class_without_category_().__mro__
        )
        FirstSharedClass = next(common_superclasses)

        codomain = FirstSharedClass(
            domain=self._right_domain.domain,
            codomain=self._left_domain.codomain[shift],
        )

        return codomain, shift

    def _act_(
        self,
        g: GradedFreeModule_Homomorphism,
        s: GradedFreeModule_Homomorphism,
    ) -> GradedFreeModule_Homomorphism:
        left = g
        right = s
        left_shifted = left[self._hom_shift]
        left_shifted_supp = left_shifted.support()
        right_supp = right.support()
        if len(left_shifted_supp) < len(right_supp):
            prod_supp = (x for x in left_shifted_supp if x in right_supp)
        else:
            prod_supp = (x for x in right_supp if x in left_shifted_supp)
        result_dict = {}
        for key in prod_supp:
            value = left_shifted(key) * right(key)
            if value:
                result_dict[key] = value

        return self._codomain._element_constructor_(
            result_dict,
            check=False,
            copy=False,
        )


class GradedFreeModule_Homset(UniqueRepresentation, Module):
    """
    Homset for graded free modules over algebras.

    Naturally a module over the original algebra.
    """

    Element = GradedFreeModule_Homomorphism
    coerce_to_superclass_parents = True
    convert_to_subclass_parents = True

    @lazy_class_attribute
    def CycleClass(cls):
        """
        Class of (co)cycles in homset, i.e. objects with zero differential.

        It is a class of chain maps.
        """
        return ComplexOfFreeModules_Homset

    def __classcall__(
        cls,
        domain: GradedFreeModule | ShiftedGradedFreeModule,
        codomain: GradedFreeModule | ShiftedGradedFreeModule,
        base=None,
    ):
        return super().__classcall__(
            cls,
            domain=domain,
            codomain=codomain,
            base=base,
        )

    def __init__(
        self,
        domain: GradedFreeModule | ShiftedGradedFreeModule,
        codomain: GradedFreeModule | ShiftedGradedFreeModule,
        base,
    ):
        ring = get_coercion_model().common_parent(domain.ring(), codomain.ring())
        assert domain.hom_grading_group() is codomain.hom_grading_group()
        self.domain = domain
        self.codomain = codomain
        self.ring = ring

        if base is None:
            base = self.default_base()
        else:
            assert ring.has_coerce_map_from(base)

        Module.__init__(
            self,
            base=base,
        )

    def default_base(self):
        return self.end_algebra.base()

    @lazy_attribute
    def end_algebra(self):
        return FreeRankOneModule_Endset(self.ring)

    def _element_constructor_(self, *args, **kwds):
        if len(args) == 1:
            if isinstance(args[0], Element):
                if args[0].parent() == self:
                    return args[0]
        return self.element_class(self, *args, **kwds)

    def hom_grading_group(self):
        return self.domain.hom_grading_group()

    def one(self):
        assert self.domain == self.codomain
        end_algebra = self.end_algebra
        matrix_dict = {
            degree: matrix.identity(
                end_algebra,
                dim,
                sparse=True,
                immutable=True,
            )
            for degree, dim in self.domain.component_rank_iter()
        }
        return self._element_constructor_(matrix_dict, check=False)

    def zero(self):
        return self._element_constructor_({}, check=False)

    def _coerce_map_from_(self, other):
        if isinstance(other, self._class_without_category_()):
            if (
                other.coerce_to_superclass_parents
                & (self.codomain == other.codomain)
                & (self.domain == other.domain)
                & (self.ring == other.ring)  # relax the condition?
            ):
                return lambda parent, x: self._element_constructor_(
                    x._map_components,
                    check=False,
                )

    def _convert_map_from_(self, other):
        if isinstance(other, GradedFreeModule_Homset):
            if isinstance(self, other._class_without_category_()):
                if (
                    other.convert_to_subclass_parents
                    & (self.codomain == other.codomain)
                    & (self.domain == other.domain)
                    & (self.ring == other.ring)  # relax the condition?
                ):
                    return lambda parent, x: self._element_constructor_(
                        x._map_components
                    )

    def _get_action_(self, other, op, self_on_left):
        if op == operator.mul:
            if self_on_left:
                if isinstance(other, GradedFreeModule_Homset):
                    return HomHomMultiplication(left_parent=self, right_parent=other)
        return super()._get_action_(other, op, self_on_left)

    @lazy_attribute
    def _self_action(self):
        return HomHomMultiplication(left_parent=self, right_parent=self)

    @classmethod
    def _class_without_category_(cls):
        """
        Return class without "with_category" part.

        For a parent class `ParentClass` Sagemath creates a new
        child class with name `ParentClass_with_category`.
        We return the preceeding class.
        """
        return cls.__mro__[1]

    def _repr_(self):
        result = "A space of graded maps between graded modules:\n"
        result += "from\n"
        result += repr(self.domain)
        result += "\nto\n"
        result += repr(self.codomain)
        result += "\n"

        return result

    def _element_check_(self, element):
        print("<0")
        pass

    def shift_domain(self, shift):
        """
        Shift domain by `shift`.
        """
        return self._replace_(
            domain=self.domain[shift],
        )

    def shift_codomain(self, shift):
        """
        Shift codomain by `shift`.
        """
        return self._replace_(
            codomain=self.codomain[shift],
        )

    def shift(self, shift):
        """
        Shift both domain and codomain by `shift`.
        """
        return self._replace_(
            domain=self.domain[shift],
            codomain=self.codomain[shift],
        )

    def shift_isomorphism(self, shift):
        self_shifted = self.shift(shift)
        return self.hom(
            self_shifted,
            lambda x: x[shift],
        )

    def apply_differential(self, morphism):
        """
        Return the image of the morphism under the differential.

        One can define it as
        `self.codomain.differential() * morphism
        - morphism * self.domain.differential()`;
        In our applications the image is often zero.
        Then it's faster to get the zero doing the computation
        grading-by grading, without storing intermediate result
        for each of the summands.
        """
        assert morphism.parent() == self
        assert isinstance(
            self.domain, ComplexOfFreeModules | ShiftedComplexOfFreeModules
        )
        assert isinstance(
            self.codomain, ComplexOfFreeModules | ShiftedComplexOfFreeModules
        )

        domain_diff = self.domain.differential()
        codomain_diff = self.codomain.differential()
        shifted_morphism = morphism[self.domain.differential_degree()]

        morphism_supp = morphism.support()
        codomain_diff_supp = codomain_diff.support()
        if len(morphism_supp) < len(codomain_diff_supp):
            first_term_supp = (x for x in morphism_supp if x in codomain_diff_supp)
        else:
            first_term_supp = (x for x in codomain_diff_supp if x in morphism_supp)
        first_term_supp = frozenset(first_term_supp)

        domain_diff_supp = domain_diff.support()
        shifted_morphism_supp = shifted_morphism.support()
        if len(shifted_morphism_supp) < len(domain_diff_supp):
            second_term_supp = (
                x for x in shifted_morphism_supp if x in domain_diff_supp
            )
        else:
            second_term_supp = (
                x for x in domain_diff_supp if x in shifted_morphism_supp
            )
        second_term_supp = frozenset(second_term_supp)

        result_dict = {}
        for grading in first_term_supp | second_term_supp:
            if grading in first_term_supp:
                if grading in second_term_supp:
                    value = codomain_diff(grading) * morphism(
                        grading
                    ) - shifted_morphism(grading) * domain_diff(grading)
                else:
                    value = codomain_diff(grading) * morphism(grading)
            else:
                value = -shifted_morphism(grading) * domain_diff(grading)
            if value:
                result_dict[grading] = value

        # the differential is automatically a (co)cycle
        cycle_module = morphism.parent().CycleClass(
            self.domain,
            self.codomain[self.codomain.differential_degree()],
        )
        return cycle_module._element_constructor_(
            result_dict,
            check=False,
        )

    def differential(self):
        from sage.structure.unique_representation import unreduce

        # making differential's codomain by shifting
        # codomain in self by differential's degree.
        shift = self.codomain.differential_degree()
        cls, args, kwrds = self._reduction
        replacements = {
            "codomain": self.codomain[shift],
        }
        new_kwrds = kwrds | replacements
        differential_codomain = unreduce(cls, args, new_kwrds)
        return self.hom(lambda x: self.apply_differential(x), differential_codomain)

    def _replace_(self, **replacements):
        """
        Make a similar parent with several adjustments.

        Compare to _replace of named tuples.
        """
        from sage.structure.unique_representation import unreduce

        cls, args, kwrds = self._reduction
        new_kwrds = kwrds | replacements
        return unreduce(cls, args, new_kwrds)


@dataclass(frozen=True, init=True, eq=True, repr=False)
class KLRWIrreducibleProjectiveModule:
    state: KLRWstate
    equivariant_degree: QuiverGradingGroupElement | int
    grading_group: InitVar[QuiverGradingGroup | None] = None

    def __post_init__(self, grading_group: QuiverGradingGroup | None):
        """
        If grading_group was given coerse grading into an element of it
        """
        if grading_group is not None:
            # bypass protection in frozen=True to change the entry
            super().__setattr__(
                "equivariant_degree",
                grading_group(self.equivariant_degree),
            )

    def __repr__(self):
        return (
            "T_"
            + self.state.__repr__()
            + "{"
            + self.equivariant_degree.__repr__()
            + "}"
        )

    def __getitem__(self, key):
        shift = self.equivariant_degree.parent()(key)

        if shift:
            return KLRWIrreducibleProjectiveModule(
                state=self.state,
                equivariant_degree=self.equivariant_degree + shift,
            )
        else:
            return self


'''
class ParameterHomMultiplication(Action):
    """
    Multiplication of a hom of graded projectives
    by a symmetric homogeneous element in the dot algebra.

    Default coercion uses the unit in KLRW which is not the best
    way to do multiplication by parameters.
    """

    def __init__(self, other, hom_parent):
        self.dot_algebra = hom_parent.KLRW_algebra().base()
        self.coerce_map = self.dot_algebra.coerce_map_from(other)
        Action.__init__(self, G=other, S=hom_parent, is_left=True, op=operator.mul)

    def codomain(self):
        raise AttributeError("Codomain depend on the dot algebra element")

    def _act_(
        self,
        g,
        h: KLRWHomOfGradedProjectivesElement,
    ) -> KLRWHomOfGradedProjectivesElement:
        g = self.coerce_map(g)
        grading_group = self.domain().KLRW_algebra().grading_group
        element_degree = self.dot_algebra.element_degree(
            g,
            grading_group,
            check_if_homogeneous=True,
        )
        assert self.dot_algebra.is_element_symmetric(g), "The element is not symmetric"

        hom_codomain_projectives = {}
        h_codomain = self.domain().codomain
        for hom_deg, projs in h_codomain.projectives.items():
            hom_codomain_projectives[hom_deg] = [
                replace(pr, equivariant_degree=pr.equivariant_degree + element_degree)
                for pr in projs
            ]

        hom_codomain = KLRWPerfectComplex(
            KLRW_algebra=self.domain().codomain.KLRW_algebra,
            differentials=self.domain().codomain.differentials,
            projectives=hom_codomain_projectives,
            degree=self.domain().codomain.degree,
            grading_group=self.domain().codomain.grading_group,
            mod2grading=self.domain().codomain.mod2grading,
            check=True,
        )

        product_parent = KLRWHomOfGradedProjectives(
            self.domain().domain,
            hom_codomain,
            shift=self.domain().shift,
        )

        elem_times_h_dict = {stype: g * elem for stype, elem in h}

        return product_parent._from_dict(elem_times_h_dict)
'''


class KLRWIrreducibleProjectiveModule_Homset:
    pass


class KLRWDirectSumOfProjectives(GradedFreeModule):
    """
    A direct sum of irreducible projectives with shifts.

    Homological shifts and equivariant shifts are treated differently.
    """

    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedKLRWDirectSumOfProjectives

    @lazy_class_attribute
    def HomsetClass(cls):
        return KLRWDirectSumOfProjectives_Homset

    @staticmethod
    def __classcall__(
        cls,
        ring: KLRWAlgebra,
        projectives: dict[
            HomologicalGradingGroupElement | ExtendedQuiverGradingGroupElement,
            Iterable[KLRWIrreducibleProjectiveModule | KLRWstate]
            | KLRWIrreducibleProjectiveModule
            | KLRWstate,
        ],
        homological_grading_names: Iterable[Any] = (None,),
        **kwargs,
    ):
        homological_grading_names = tuple(homological_grading_names)
        extended_grading_group = cls._extended_grading_group_(
            ring, homological_grading_names
        )

        _projectives = defaultdict(list)
        if not isinstance(projectives, dict):
            projectives = dict(projectives)
        for grading, value in projectives.items():
            if isinstance(value, KLRWIrreducibleProjectiveModule) or isinstance(
                value, KLRWstate
            ):
                value = [value]
            else:
                value = list(value)

            if isinstance(grading, ExtendedQuiverGradingGroupElement):
                grading = extended_grading_group(grading)
                grading_hom_part = grading.homological_part()
                grading_eq_part = grading.equivariant_part()
            else:
                try:
                    if isinstance(grading, int):
                        grading = ZZ(grading)
                    grading = extended_grading_group.homological_part(grading)
                except TypeError:
                    raise ValueError(
                        "Unkown type of grading: {}".format(grading.__class__)
                    )
                grading_hom_part = grading
                grading_eq_part = ring.grading_group()

            for i in range(len(value)):
                if isinstance(value[i], KLRWstate):
                    value[i] = KLRWIrreducibleProjectiveModule(
                        state=value[i],
                        equivariant_degree=grading_eq_part,
                        grading_group=ring.grading_group,
                    )
                elif isinstance(value[i], KLRWIrreducibleProjectiveModule):
                    value[i] = KLRWIrreducibleProjectiveModule(
                        state=value[i].state,
                        equivariant_degree=value[i].equivariant_degree
                        + grading_eq_part,
                        grading_group=ring.grading_group,
                    )

            _projectives[grading_hom_part] += value

        # make immutable
        projectives = {hom_deg: tuple(projs) for hom_deg, projs in _projectives.items()}
        # make hashable
        projectives = frozenset(projectives.items())

        return UniqueRepresentation.__classcall__(
            cls,
            ring=ring,
            projectives=projectives,
            homological_grading_names=homological_grading_names,
            **kwargs,
        )

    def __init__(
        self,
        ring: KLRWAlgebra,
        projectives: frozenset[
            tuple[HomologicalGradingGroupElement, KLRWIrreducibleProjectiveModule]
        ],
        homological_grading_names,
    ):
        self._KLRW_algebra = ring
        self._projectives = MappingProxyType(
            {grading: projs for grading, projs in projectives}
        )
        self._extended_grading_group = self._extended_grading_group_(
            ring,
            homological_grading_names,
        )

    @staticmethod
    def _extended_grading_group_(ring, homological_grading_names):
        homological_grading_group = HomologicalGradingGroup(
            homological_grading_names=homological_grading_names
        )
        return ExtendedQuiverGradingGroup(
            equivariant_grading_group=ring.grading_group,
            homological_grading_group=homological_grading_group,
        )

    def KLRW_algebra(self):
        return self._KLRW_algebra

    def projectives(self, grading=None, position=None):
        if grading is None:
            assert position is None
            return self._projectives
        grading = self.hom_grading_group()(grading)
        if grading in self._projectives:
            if position is None:
                return self._projectives[grading]
            else:
                return self._projectives[grading][position]

        return tuple()

    def projectives_iter(self):
        return self._projectives.items()

    def gradings(self):
        return self._projectives.keys()

    def component_rank(self, grading):
        return len(self.projectives(grading))

    def component_rank_iter(self):
        for grading, projs in self._projectives.items():
            yield (grading, len(projs))

    def equivariant_grading_group(self):
        return self._extended_grading_group.equivariant_part

    def shift_group(self):
        return self._extended_grading_group

    def hom_grading_group(self):
        return self._extended_grading_group.homological_part

    def ring(self):
        return self.KLRW_algebra()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            shift = self.shift_group()(*key)
        else:
            shift = self.shift_group()(key)

        if shift:
            return self.ShiftedClass(
                self,
                shift=shift,
            )
        else:
            return self

    def __repr__(self):
        from pprint import pformat

        result = "A direct sum of projectives\n"
        result += pformat(dict(self._projectives))

        return result

    def base_change(self, other: Ring):
        assert other.has_coerce_map_from(self.ring())
        return self._replace_(ring=other)


class ShiftedKLRWDirectSumOfProjectives(ShiftedGradedFreeModule):
    """
    A graded module with shifted gradings.
    """

    @lazy_class_attribute
    def OriginalClass(cls):
        return KLRWDirectSumOfProjectives

    def __post_init__(self):
        assert isinstance(self.original, KLRWDirectSumOfProjectives)

    def KLRW_algebra(self):
        return self.original.KLRW_algebra()

    def equivariant_grading_group(self):
        return self.original.equivariant_grading_group()

    def shift_group(self):
        return self.original.shift_group()

    def equivariant_shift(self):
        return self.shift.equivariant_part()

    def projectives(self, grading=None, position=None):
        if grading is None:
            assert position is None
            return dict(self.projectives_iter())
        grading = self.hom_grading_group()(grading)

        hom_shift = self.hom_shift()
        if position is None:
            projs = self.original.projectives(
                grading + hom_shift,
            )
            return self._shift_tuple_of_projectives_(projs)
        else:
            proj = self.original.projectives(grading + hom_shift, position)
            equivariant_shift = self.equivariant_shift()
            return proj[equivariant_shift]

    def _shift_tuple_of_projectives_(
        self, projs: tuple[KLRWIrreducibleProjectiveModule]
    ):
        equivariant_shift = self.equivariant_shift()
        if equivariant_shift:
            return tuple(pr[equivariant_shift] for pr in projs)
        else:
            return projs

    def projectives_iter(self):
        hom_shift = self.hom_shift()
        for grading, projs in self.original.projectives_iter():
            yield (
                grading - hom_shift,
                self._shift_tuple_of_projectives_(projs),
            )

    def gradings(self):
        hom_shift = self.hom_shift()
        return frozenset(grading - hom_shift for grading in self.original.gradings())

    def component_rank(self, grading):
        hom_shift = self.hom_shift()
        return self.original.component_rank(grading + hom_shift)

    def component_rank_iter(self):
        hom_shift = self.hom_shift()
        for grading, dim in self.original.component_rank_iter():
            yield (grading - hom_shift, dim)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            extra_shift = self.shift_group()(*key)
        else:
            extra_shift = self.shift_group()(key)
        return self.shift_object(extra_shift)


class KLRWDirectSumOfProjectives_Homomorphism(GradedFreeModule_Homomorphism):
    # by default, -matrix is done by using (-1) in the base ring.
    # It's too slow because KLRW has a 1 with many pieces.
    # Here we do a workaround, by explicitly making
    # the matrix from negatives of entries
    @staticmethod
    def _negate_matrix_(mat):
        neg_mat_dict = {key: -value for key, value in mat.dict(copy=False).items()}
        # constructor will copy the dict,
        # but it does not accept iterators.
        # is there any way to do it without double-copying?
        return matrix(
            mat.base_ring(),
            nrows=mat.nrows(),
            ncols=mat.ncols(),
            entries=neg_mat_dict,
            sparse=True,
            immutable=True,
        )

    def _neg_(self):
        neg_map_components = {
            grading: self._negate_matrix_(mat)
            for grading, mat in self._map_components.items()
        }

        return self.__class__(
            self.parent(),
            neg_map_components,
            check=False,
            copy=False,
        )

    def __call__(self, grading, sign=False):
        """
        Return a component of the homomorphism.
        """
        component = super().__call__(grading, sign=False)
        if sign and component is not None:
            component = self._negate_matrix_(component)

        return component


class KLRWDirectSumOfProjectives_Homset(GradedFreeModule_Homset):
    Element = KLRWDirectSumOfProjectives_Homomorphism

    @lazy_class_attribute
    def CycleClass(cls):
        """
        Class of (co)cycles in homset, i.e. objects with zero differential.

        It is a class of chain maps.
        """
        return KLRWPerfectComplex_Homset

    @lazy_class_attribute
    def BasisClass(cls):
        """
        Class of (co)cycles in homset, i.e. objects with zero differential.

        It is a class of chain maps.
        """
        return KLRWDirectSumOfProjectives_HomsetBasis

    def KLRW_algebra(self):
        return self.ring

    def default_base(self):
        return self.KLRW_algebra().scalars()

    def _element_check_(self, element):
        print("<1")
        isomorphism_to_algebra = self.end_algebra.isomorphism_to_algebra
        for degree, mat in element:
            domain_projs = self.domain.projectives(degree)
            codomain_projs = self.codomain.projectives(degree)
            for (i, j), elem in mat.dict(copy=False).items():
                elem = isomorphism_to_algebra(elem)
                codomain_pr = codomain_projs[i]
                domain_pr = domain_projs[j]

                right_state = elem.right_state(check_if_all_have_same_right_state=True)
                assert codomain_pr.state == right_state
                left_state = elem.left_state(check_if_all_have_same_left_state=True)
                assert domain_pr.state == left_state
                degree = elem.degree(check_if_homogeneous=True)
                assert (
                    codomain_pr.equivariant_degree - domain_pr.equivariant_degree
                    == degree
                ), (
                    repr(codomain_pr.equivariant_degree)
                    + " "
                    + repr(domain_pr.equivariant_degree)
                    + " "
                    + repr(degree)
                )

        super()._element_check_(element)

    def one(self):
        assert self.domain == self.codomain
        end_algebra = self.end_algebra
        matrix_dict = {
            degree: matrix(
                end_algebra,
                ncols=len(projs),
                nrows=len(projs),
                entries={
                    (i, i): self.ring.KLRWmonomial(state=pr.state)
                    for i, pr in enumerate(projs)
                },
                sparse=True,
                immutable=True,
            )
            for degree, projs in self.domain.projectives_iter()
        }
        return self._element_constructor_(matrix_dict, check=False)

    @lazy_attribute
    def basis(self):
        return self.BasisClass(self)

    def _repr_(self):
        result = "A space of graded maps between sums of KLRW projective modules:\n"
        result += "from\n"
        result += repr(self.domain)
        result += "\nto\n"
        result += repr(self.codomain)
        result += "\n"

        return result


class KLRWDirectSumOfProjectives_HomsetBasis(UniqueRepresentation):
    def __init__(self, homset: KLRWDirectSumOfProjectives_Homset):
        assert isinstance(homset, KLRWDirectSumOfProjectives_Homset)
        self._homset = homset

    @dataclass(frozen=True)
    class EntryType:
        """
        Records combinatorial data about
        Hom(P_i[grading],P_j[grading+shift])
        """

        grading: int  # cohomological grading
        domain_index: int
        codomain_index: int

    @lazy_attribute
    def _as_tuple_(self):
        return tuple(self)

    def __call__(self, index=None):
        if index is not None:
            return self._as_tuple_[index]
        return self._as_tuple_

    def __iter__(self):
        for type_ in self._types_iter_():
            graded_component = self.graded_component_of_type(type_)
            for elem in graded_component.basis():
                indices = (type_.codomain_index, type_.domain_index)
                yield self._homset._element_constructor_(
                    {type_.grading: {indices: elem}},
                    check=False,
                )

    @lazy_attribute
    def _types_(self):
        return tuple(self._types_iter_())

    def types(self, index=None):
        if index is not None:
            return self._types_[index]
        return self._types_

    def _types_iter_(self):
        domain_gradings = self._homset.domain.gradings()
        codomain_gradings = self._homset.codomain.gradings()
        if len(domain_gradings) < len(codomain_gradings):
            gradings = (x for x in domain_gradings if x in codomain_gradings)
        else:
            gradings = (x for x in codomain_gradings if x in domain_gradings)

        # using sorted() adds consistency of the output
        # otherwise the order of the blocks may vary
        # UniqueRepresentation & hashing takes care of most accidental swaps of blocks
        # However we have issues when identical complexes are defined
        # with different input data [e.g. different classes]
        for grading in sorted(gradings):
            matrix_index_iter = product(
                range(len(self._homset.codomain.projectives(grading))),
                range(len(self._homset.domain.projectives(grading))),
            )
            for i, j in matrix_index_iter:
                type_ = self.EntryType(
                    grading=grading,
                    codomain_index=i,
                    domain_index=j,
                )
                dim = self.graded_component_of_type(type_).dimension()
                if dim > 0:
                    yield type_

    @lazy_attribute
    def _subdivisions_(self):
        """
        The vector over base ring encoding of the elements has block structure.
        Here we record the indices where the borders between the blocks are.
        If index is given, returns _subdivisions_()[index]
        """
        subdivisions = [0]
        for t in self.types():
            dim = self.graded_component_of_type(t).dimension()
            subdivisions.append(subdivisions[-1] + dim)
        return tuple(subdivisions)

    def subdivisions(self, index=None):
        """
        The vector over base ring encoding of the elements has block structure.
        Here we record the indices where the borders between the blocks are.
        If index is given, returns _subdivisions_()[index]
        """
        if index is not None:
            return self._subdivisions_[index]
        return self._subdivisions_

    def __len__(self):
        return self.subdivisions(-1)

    @lazy_attribute
    def coordinate_free_module(self):
        return FreeModule(self._homset.KLRW_algebra().scalars(), len(self), sparse=True)

    def graded_component_of_type(self, type_):
        left = self._homset.domain.projectives(type_.grading, type_.domain_index)
        right = self._homset.codomain.projectives(type_.grading, type_.codomain_index)
        left_state = left.state
        right_state = right.state
        degree = right.equivariant_degree - left.equivariant_degree
        return self._homset.KLRW_algebra()[left_state:right_state:degree]

    @lazy_attribute
    def _type_to_subdivision_index_(self) -> MappingProxyType:
        """
        For a type gives the index where the corresponding subdivision starts.
        """
        result = {}
        for i, type_ in enumerate(self.types()):
            result[type_] = self.subdivisions(i)
        return MappingProxyType(result)

    def _element_from_vector_(self, vect, check=True):
        from bisect import bisect_right

        assert len(vect) == self.subdivisions(-1)

        # We optimize for vect representing a sparse element,
        # So the length is much less than self._subdivisions_
        result_dict = {}
        for i, value in vect.dict(copy=False).items():
            type_ind = bisect_right(self._subdivisions_, i) - 1
            rel_index = i - self._subdivisions_[type_ind]

            type_ = self.types(type_ind)
            graded_component = self.graded_component_of_type(type_)
            entry = graded_component.basis(as_tuples=False)[rel_index]
            entry *= value
            indices = (type_.codomain_index, type_.domain_index)
            if type_.grading in result_dict:
                res_mat_dict = result_dict[type_.grading]
                if indices in res_mat_dict:
                    res_mat_dict[indices] += entry
                else:
                    res_mat_dict[indices] = entry
            else:
                result_dict[type_.grading] = {indices: entry}

        return self._homset._element_constructor_(result_dict, check=check)

    def _vector_from_element_(self, elem, immutable=True):
        assert isinstance(elem, KLRWDirectSumOfProjectives_Homomorphism)
        isomorphism_to_algebra = self._homset.end_algebra.isomorphism_to_algebra
        # create a mutable zero vector
        vect = self.coordinate_free_module()
        for grading, mat in elem:
            for (i, j), coeff in mat.dict(copy=False).items():
                type_ = self.EntryType(
                    grading=grading,
                    domain_index=j,
                    codomain_index=i,
                )
                graded_component = self.graded_component_of_type(type_)
                begin = self._type_to_subdivision_index_[type_]
                # coeff is an element of the opposite algebra;
                # extract the original algebra element from it.
                coeff = isomorphism_to_algebra(coeff)
                vector_part = graded_component._vector_from_element_(coeff)
                for rel_ind, scalar in vector_part.dict(copy=False).items():
                    vect[begin + rel_ind] = scalar

        if immutable:
            vect.set_immutable()
        return vect

    @cached_method
    def _differential_matrix_(self, keep_subdivisions=True):
        """
        Differential matrix acting on `self._homset`.

        Makes a matrix of the differential that acts *on* this
        space of graded morphisms and giving a hom of type
        domain->codomain[differential_degree]
        using `self` [and similar basis in the target] as basis.
        """
        assert isinstance(
            self._homset.domain, ComplexOfFreeModules | ShiftedComplexOfFreeModules
        )
        assert isinstance(
            self._homset.codomain, ComplexOfFreeModules | ShiftedComplexOfFreeModules
        )
        dot_algebra = self._homset.KLRW_algebra().base()
        grading_group = self._homset.KLRW_algebra().grading_group
        assert dot_algebra.no_parameters_of_zero_degree(grading_group), (
            "There are zero-degree parameters in the dot algebra.\n"
            + "Set them to scalars or extend the grading group."
        )

        isomorphism_to_algebra = self._homset.end_algebra.isomorphism_to_algebra
        # basis of the next homset [with respect to the differential action]
        next_homset = self._homset.shift_codomain(
            self._homset.codomain.differential_degree()
        )
        next_ = next_homset.basis

        diff_mat = matrix(
            self._homset.KLRW_algebra().scalars(),
            len(next_),
            len(self),
            sparse=True,
        )
        if keep_subdivisions:
            diff_mat._subdivisions = (
                list(next_.subdivisions()),
                list(self.subdivisions()),
            )

        # matrix of left multiplication by d.
        for type_ in self.types():
            column_subdivision_index = self._type_to_subdivision_index_[type_]
            grading = type_.grading
            klrw_mat = self._homset.codomain.differential(grading)
            if klrw_mat is not None:
                for (i, j), d_entry in klrw_mat.dict(copy=False).items():
                    if j == type_.codomain_index:
                        product_type = self.EntryType(
                            grading=type_.grading,
                            domain_index=type_.domain_index,
                            codomain_index=i,
                        )
                        # product type may be missing if the dimension of the
                        # corresponding graded component is zero.
                        if product_type in next_._type_to_subdivision_index_:
                            row_subdivision_index = next_._type_to_subdivision_index_[
                                product_type
                            ]
                            graded_component = self.graded_component_of_type(type_)
                            # acting_on_left=False is because
                            # [despite d acts on the left]
                            # matrix coefficients multiply from the right
                            # Note that we use the algebra element,
                            # not the opposite algebra element.
                            d_entry = isomorphism_to_algebra(d_entry)
                            submatrix = d_entry.as_matrix_in_graded_component(
                                graded_component, acting_on_left=False
                            )

                            diff_mat.set_block(
                                row_subdivision_index,
                                column_subdivision_index,
                                submatrix,
                            )

        # matrix of right multiplication by d.
        # there is a sign -1
        diff_deg = self._homset.domain.differential_degree().homological_part()
        for type_ in self.types():
            column_subdivision_index = self._type_to_subdivision_index_[type_]
            grading = type_.grading - diff_deg
            klrw_mat = self._homset.domain.differential(grading)
            if klrw_mat is not None:
                for (i, j), d_entry in klrw_mat.dict(copy=False).items():
                    if i == type_.domain_index:
                        product_type = self.EntryType(
                            grading=grading,
                            domain_index=j,
                            codomain_index=type_.codomain_index,
                        )
                        # product type may be missing if the dimension of the
                        # corresponding graded component is zero.
                        if product_type in next_._type_to_subdivision_index_:
                            row_subdivision_index = next_._type_to_subdivision_index_[
                                product_type
                            ]
                            graded_component = self.graded_component_of_type(type_)
                            # acting_on_left=True is because
                            # [despite d acts on the right]
                            # matrix coefficients multiply from the right
                            # Note that we use the algebra element,
                            # not the opposite algebra element.
                            d_entry = isomorphism_to_algebra(d_entry)
                            submatrix = d_entry.as_matrix_in_graded_component(
                                graded_component, acting_on_left=True
                            )

                            for (a, b), scalar in submatrix.dict(copy=False).items():
                                diff_mat[
                                    row_subdivision_index + a,
                                    column_subdivision_index + b,
                                ] += -scalar

        return diff_mat

    @cached_method
    def _previous_differential_matrix_(self, keep_subdivisions=True):
        # basis of the next homset [with respect to the differential action]
        previous_homset = self._homset.shift_codomain(
            -self._homset.codomain.differential_degree()
        )
        return previous_homset.basis._differential_matrix_(keep_subdivisions)

    def __repr__(self):
        result = "A basis in "
        result += repr(self._homset)

        return result


class ComplexOfFreeModules(GradedFreeModule):
    """
    A complex of graded free modules.
    """

    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedComplexOfFreeModules

    @lazy_class_attribute
    def HomsetClass(cls):
        return ComplexOfFreeModules_Homset

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        assert issubclass(cls.HomsetClass, GradedFreeModule_Homset)
        return GradedFreeModule_Homset

    @staticmethod
    def __classcall__(
        cls,
        ring,
        differential: (
            dict[
                HomologicalGradingGroupElement | ExtendedQuiverGradingGroupElement,
                Matrix | dict,
            ]
            | GradedFreeModule_Homomorphism
        ),
        differential_degree,
        coerce_coefficients=True,
        **kwargs,
    ):
        # to make hashable
        if not isinstance(differential, dict):
            from klrw.cython_exts.sparse_csc import CSC_Mat
            from klrw.cython_exts.sparse_csr import CSR_Mat

            if not (
                isinstance(differential, CSR_Mat) or isinstance(differential, CSC_Mat)
            ):
                differential = dict(differential)
        if coerce_coefficients:
            new_end_algebra = FreeRankOneModule_Endset(ring)
            differential = {
                degree: mat.change_ring(new_end_algebra)
                for degree, mat in differential.items()
            }
        for mat in differential.values():
            mat.set_immutable()
        differential = frozenset(differential.items())

        return super().__classcall__(
            cls,
            ring=ring,
            differential=differential,
            differential_degree=differential_degree,
            **kwargs,
        )

    def __init__(
        self,
        differential,
        differential_degree,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(differential_degree, int) or isinstance(
            differential_degree, Integer
        ):
            differential_degree = self.hom_grading_group()(differential_degree)
        self._differential_degree = self.shift_group()(differential_degree)
        assert self._differential_degree.sign() == -1
        # We want to make differential into a morphism.
        # The correct homset is the one that does all the checks
        # except commuting with the differential
        # [even thought it is true when the differential
        # squares to zero, we can't check it before we
        # define self._differential]
        codomain = self[self._differential_degree]
        graded_hom_set = self.HomsetClassWithoutDifferential(self, codomain)
        self._differential = graded_hom_set(differential)
        diff_square = self._differential * self._differential
        assert diff_square.is_zero(), "Differential has non-zero square,\n" + repr(
            diff_square
        )
        # now we can set the differential to be a chain map; no extra checks needed.
        self._differential = self.hom(
            codomain,
            self._differential,
            check=False,
            copy=False,
        )

    def differential(self, grading=None, sign=False):
        if grading is None:
            if sign:
                return -self._differential
            else:
                return self._differential
        grading = self.hom_grading_group()(grading)
        if grading in self._differential.support():
            return self._differential(grading, sign)

        return None

    def differential_iter(self):
        return self._differential.items()

    def differential_support(self):
        return self._differential.support()

    def differential_degree(self):
        return self._differential_degree

    def differential_hom_degree(self):
        return self.hom_grading_group()(self._differential_degree)

    def __repr__(self):
        from pprint import pformat

        result = "A complex of free modules with graded dimension\n"
        result += pformat(dict(self._component_ranks))
        result += "\nand a differential\n"
        for grading in sorted(self._differential.support()):
            result += repr(grading) + " -> " + repr(grading + self._differential_degree)
            result += ":\n"
            result += repr(self._differential(grading)) + "\n"

        return result


class ShiftedComplexOfFreeModules(ShiftedGradedFreeModule):
    """
    A graded module with shifted gradings.
    """

    @lazy_class_attribute
    def OriginalClass(cls):
        return ComplexOfFreeModules

    def __post_init__(self):
        assert isinstance(self.original, ComplexOfFreeModules)

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        return cls.OriginalClass.HomsetClassWithoutDifferential

    def differential(self, grading=None):
        if grading is None:
            hom_set = self.original.differential().parent()
            shifted_homset = hom_set.shift(self.shift)
            return shifted_homset._element_constructor_(
                self.differential_iter(),
                check=False,
            )
        grading = self.hom_grading_group()(grading)

        sign = self.shift.sign()
        hom_shift = self.hom_shift()
        diff_component = self.original.differential(
            grading + hom_shift,
            sign != 1,
        )
        if diff_component is None:
            return None
        return diff_component

    def differential_support(self):
        hom_shift = self.hom_shift()
        return frozenset(
            grading - hom_shift for grading in self.original.differential_support()
        )

    def differential_iter(self):
        hom_shift = self.hom_shift()
        sign = self.shift.sign()
        if sign != 1:
            for grading, mat in self.original.differential_iter():
                yield (
                    grading - hom_shift,
                    -mat,
                )
        else:
            for grading, mat in self.original.differential_iter():
                yield (
                    grading - hom_shift,
                    mat,
                )

    def differential_degree(self):
        return self.original._differential_degree

    def differential_hom_degree(self):
        return self.hom_grading_group()(self.differential_degree())


class ComplexOfFreeModules_Homomorphism(GradedFreeModule_Homomorphism):
    def _cone_component_ranks(self):
        parent = self.parent()
        diff_degree = parent.domain.differential_degree()
        domain_shifted = parent.domain[diff_degree]
        codomain = parent.codomain

        ranks = defaultdict(int)
        for grading, rk in domain_shifted.component_rank_iter():
            ranks[grading] += rk
        for grading, rk in codomain.component_rank_iter():
            ranks[grading] += rk

        return MappingProxyType(ranks)

    def _cone_differential(self, keep_subdivisions=True):
        parent = self.parent()
        diff_degree = parent.domain.differential_degree()
        morphism_shifted = self[diff_degree]
        domain_shifted = parent.domain[diff_degree]
        codomain = parent.codomain

        gradings = frozenset(domain_shifted.differential_support())
        gradings |= frozenset(codomain.differential_support())
        gradings |= frozenset(morphism_shifted.support())

        diff_hom_degree = parent.domain.differential_hom_degree()
        differential = {}
        for grading in gradings:
            next_grading = grading + diff_hom_degree
            # the matrix has block structure
            # columns are splitted into two categories by left_block_size
            left_block_size = domain_shifted.component_rank(grading)
            # the total number of columns
            new_domain_rk = left_block_size + codomain.component_rank(grading)
            # rows are splitted into two categories by top_block_size
            top_block_size = domain_shifted.component_rank(next_grading)
            # the total number of rows
            new_codomain_rk = top_block_size + codomain.component_rank(next_grading)

            differential_component = matrix(
                parent.end_algebra,
                ncols=new_domain_rk,
                nrows=new_codomain_rk,
                sparse=True,
            )

            top_left_block = domain_shifted.differential(grading)
            if top_left_block is not None:
                differential_component.set_block(0, 0, top_left_block)
            bottom_left_block = morphism_shifted(grading)
            if bottom_left_block is not None:
                differential_component.set_block(top_block_size, 0, bottom_left_block)
            bottom_right_block = codomain.differential(grading)
            if bottom_right_block is not None:
                differential_component.set_block(
                    top_block_size, left_block_size, bottom_right_block
                )
            if keep_subdivisions:
                differential_component._subdivisions = (
                    [0, top_block_size, new_codomain_rk],
                    [0, left_block_size, new_domain_rk],
                )

            differential_component.set_immutable()
            differential[grading] = differential_component

        return differential

    def cone(self, keep_subdivisions=True):
        return ComplexOfFreeModules(
            component_ranks=self._cone_component_ranks(),
            ring=self.parent().ring,
            differential_degree=self.parent().domain.differential_degree(),
            differential=self._cone_differential(),
        )


class ComplexOfFreeModules_Homset(GradedFreeModule_Homset):
    Element = ComplexOfFreeModules_Homomorphism

    @lazy_class_attribute
    def HomologyClass(cls):
        return KLRWPerfectComplex_Ext0set

    def __init__(
        self,
        domain: GradedFreeModule,
        codomain: GradedFreeModule,
        **kwargs,
    ):
        assert domain.differential_degree() == codomain.differential_degree()

        super().__init__(
            domain=domain,
            codomain=codomain,
            **kwargs,
        )

    def _element_check_(self, element):
        print("<2")
        image = self.apply_differential(element)
        assert image.is_zero(), "Morphism does not commute with differential."

        super()._element_check_(element)

    def _convert_map_from_(self, other):
        if isinstance(other, self.HomologyClass):
            if (
                (self.codomain == other.codomain)
                & (self.domain == other.domain)
                & (self.ring == other.ring)  # relax the condition?
            ):
                return lambda parent, x: self._element_constructor_(
                    self,
                    x._map_components,
                    check=False,
                )

        return super()._convert_map_from_(other)

    def _repr_(self):
        result = "A space of chain maps between complexes of free modules:\n"
        result += "from\n"
        result += repr(self.domain)
        result += "\nto\n"
        result += repr(self.codomain)
        result += "\n"

        return result


class KLRWPerfectComplex(ComplexOfFreeModules, KLRWDirectSumOfProjectives):
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedKLRWPerfectComplex

    @lazy_class_attribute
    def HomsetClass(cls):
        return KLRWPerfectComplex_Homset

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        assert issubclass(cls.HomsetClass, KLRWDirectSumOfProjectives_Homset)
        return KLRWDirectSumOfProjectives_Homset

    def __repr__(self):
        from pprint import pformat

        result = "A complex of projective modules modules\n"
        result += pformat(dict(self.projectives()))
        result += "\nand a differential\n"
        for grading in sorted(self.differential().support()):
            result += repr(grading) + " -> "
            result += repr(grading + self.differential_hom_degree())
            result += ":\n"
            result += repr(self.differential(grading)) + "\n"

        return result


class ShiftedKLRWPerfectComplex(
    ShiftedComplexOfFreeModules, ShiftedKLRWDirectSumOfProjectives
):
    @lazy_class_attribute
    def OriginalClass(cls):
        return KLRWPerfectComplex


class KLRWPerfectComplex_Homomorphism(
    ComplexOfFreeModules_Homomorphism, KLRWDirectSumOfProjectives_Homomorphism
):
    pass


class KLRWPerfectComplex_Homset(
    ComplexOfFreeModules_Homset, KLRWDirectSumOfProjectives_Homset
):
    @lazy_class_attribute
    def BasisClass(cls):
        """
        Class of (co)cycles in homset, i.e. objects with zero differential.

        It is a class of chain maps.
        """
        return KLRWPerfectComplex_HomsetBasis

    Element = KLRWPerfectComplex_Homomorphism

    @lazy_class_attribute
    def HomologyClass(cls):
        """
        Class of (co)cycles in homset, i.e. objects with zero differential.

        It is a class of chain maps.
        """
        return KLRWPerfectComplex_Ext0set

    def homology(self):
        return self.HomologyClass(
            self.domain,
            self.codomain,
        )


class KLRWPerfectComplex_HomsetBasis(UniqueRepresentation):
    def __init__(self, homset: KLRWPerfectComplex_Homset):
        assert isinstance(homset, KLRWPerfectComplex_Homset)
        self._homset = homset
        self._ambient_homset = homset.domain.HomsetClassWithoutDifferential(
            domain=homset.domain,
            codomain=homset.codomain,
        )

    @lazy_attribute
    def _as_tuple_(self):
        return tuple(self)

    def __call__(self, index=None):
        if index is not None:
            return self._as_tuple_[index]
        return self._as_tuple_

    def __iter__(self):
        for vec in self.coordinate_free_module.basis():
            yield self._element_from_vector_(vec, check=False)

    def __len__(self):
        return self.coordinate_free_module.rank()

    @lazy_attribute
    def next_differential(self):
        return self._ambient_homset.basis._differential_matrix_()

    @lazy_attribute
    def previous_differential(self):
        return self._ambient_homset.basis._previous_differential_matrix_()

    @lazy_attribute
    def coordinate_free_module(self):
        differential = self.next_differential
        kernel = differential.right_kernel_matrix().sparse_matrix().row_module()
        return kernel

    def _element_from_vector_(self, vect, check=True):
        ambient_element = self._ambient_homset.basis._element_from_vector_(
            vect,
            check=check,
        )
        return self._homset._element_constructor_(ambient_element, check=check)

    def _vector_from_element_(self, elem, check=True, immutable=True):
        vec = self._ambient_homset.basis._vector_from_element_(elem, immutable)
        if check:
            image = self.next_differential * vec
            assert image.is_zero(), "Element {} is not a cycle".format(elem)
        return vec


class KLRWPerfectComplex_Ext0(
    KLRWDirectSumOfProjectives_Homomorphism,
    ComplexOfFreeModules_Homomorphism,
):
    def representative(self):
        return self.parent().representative(self)

    def in_coordinates(self):
        return self.parent().in_coordinates(self)

    def _richcmp_(self, right, op):
        from sage.structure.richcmp import richcmp

        return richcmp(self.in_coordinates(), right.in_coordinates(), op)

    def __hash__(self):
        return hash(self.in_coordinates())

    def _cone_projectives(self):
        parent = self.parent()
        diff_degree = parent.domain.differential_degree()
        domain_shifted = parent.domain[diff_degree]
        codomain = parent.codomain

        projectives = defaultdict(list)
        for grading, projs in domain_shifted.projectives_iter():
            projectives[grading] += projs
        for grading, projs in codomain.projectives_iter():
            projectives[grading] += projs

        return MappingProxyType(projectives)

    def cone(self, keep_subdivisions=True):
        return KLRWPerfectComplex(
            projectives=self._cone_projectives(),
            ring=self.parent().ring,
            differential_degree=self.parent().domain.differential_degree(),
            differential=self._cone_differential(),
        )


class KLRWPerfectComplex_Ext0set(KLRWPerfectComplex_Homset):
    """
    Ext^0 for complexes of projective modules over KLRW algebras.
    """

    Element = KLRWPerfectComplex_Ext0
    coerce_to_superclass_parents = False
    convert_to_subclass_parents = False

    @lazy_class_attribute
    def BasisClass(cls):
        """
        Class of (co)cycles in homset, i.e. objects with zero differential.

        It is a class of chain maps.
        """
        return KLRWPerfectComplex_Ext0setGenerators

    @lazy_attribute
    def cycle_module(self):
        """
        Ambient cycle module.

        Chain maps between same domain and codomain.
        """
        return self.CycleClass(
            domain=self.domain,
            codomain=self.codomain,
        )

    def one(self):
        one_as_cycle = self.CycleClass.one()
        return self.coerce(one_as_cycle)

    def zero(self):
        zero_as_cycle = self.CycleClass.zero()
        return self.coerce(zero_as_cycle)

    def _coerce_map_from_(self, other):
        if isinstance(other, self.CycleClass):
            if (
                (self.codomain == other.codomain)
                & (self.domain == other.domain)
                & (self.ring == other.ring)  # relax the condition?
            ):
                return lambda parent, x: self.class_of(x)

    def _get_action_(self, other, op, self_on_left):
        if op == operator.mul:
            if self_on_left:
                if isinstance(other, KLRWPerfectComplex_Ext0set):
                    return HomHomMultiplication(left_parent=self, right_parent=other)
        return super()._get_action_(other, op, self_on_left)

    @lazy_attribute
    def _self_action(self):
        return HomHomMultiplication(left_parent=self, right_parent=self)

    def in_coordinates(self, element):
        return self.basis._vector_from_element_(element)

    def representative(self, element):
        return self.cycle_module._element_constructor_(
            element._map_components,
            check=False,
        )

    def class_of(self, element):
        return self._element_constructor_(
            element._map_components,
            check=False,
        )

    def _repr_(self):
        result = "Ext^0 between complexes:\n"
        result += "from\n"
        result += repr(self.domain)
        result += "\nto\n"
        result += repr(self.codomain)
        result += "\n"

        return result

    @lazy_attribute
    def gens(self):
        """
        If we work over PID, this is more correcly called generators.
        """
        return self.basis


class KLRWPerfectComplex_Ext0setGenerators(UniqueRepresentation):
    def __classcall_private__(self, extset: KLRWPerfectComplex_Ext0set):
        if extset.base_ring().is_field():
            return KLRWPerfectComplex_Ext0setBasisOverField(extset)
        # there is no test on being a PID in Sage;
        # so we just check if the base is a UDF.
        elif extset.base_ring().is_unique_factorization_domain():
            return KLRWPerfectComplex_Ext0setGeneratorsOverPID(extset)
        else:
            raise ValueError("In Exts ring of definition has to be a field or a PID")

    def __init__(self, extset: KLRWPerfectComplex_Ext0set):
        assert isinstance(extset, KLRWPerfectComplex_Ext0set)
        self._extset = extset
        self._cycleset = extset.CycleClass(
            domain=extset.domain,
            codomain=extset.codomain,
        )
        self._coordinate_cycle_module = self._cycleset.basis.coordinate_free_module
        differential = self._cycleset.basis.previous_differential
        self._coordinate_boundary_module = differential.sparse_matrix().column_module()

    @lazy_attribute
    def _as_tuple_(self):
        return tuple(self)

    def __call__(self, index=None):
        if index is not None:
            return self._as_tuple_[index]
        return self._as_tuple_

    def __iter__(self):
        for gen in self.coordinate_module.gens():
            yield self._element_from_vector_(gen, check=False)

    @lazy_attribute
    def coordinate_module(self):
        homology = self._coordinate_cycle_module / self._coordinate_boundary_module
        return homology

    def _coordinate_quotient(self, lift_elem_in_coords):
        return self.coordinate_module(lift_elem_in_coords)

    def _element_from_vector_(self, vect, check=True):
        lift_vect = self._coordinate_lift(vect)
        ambient_element = self._cycleset.basis._element_from_vector_(
            lift_vect,
            check=check,
        )
        return self._extset.coerce(ambient_element)

    def _vector_from_element_(self, elem, immutable=True):
        rep = elem.representative()
        vec = self._cycleset.basis._vector_from_element_(rep)
        # `vec` is an element of a the space of all homs.
        # We need to make it an element of the cycle submodule.
        vec = self._coordinate_cycle_module(vec)
        vec = self._coordinate_quotient(vec)
        if immutable:
            vec.set_immutable()
        return vec


class KLRWPerfectComplex_Ext0setBasisOverField(KLRWPerfectComplex_Ext0setGenerators):
    def __len__(self):
        return self.coordinate_module.dimension()

    def _coordinate_lift(self, quot_elem_in_coords):
        lift = self.coordinate_module.lift(quot_elem_in_coords)
        return lift


class KLRWPerfectComplex_Ext0setGeneratorsOverPID(KLRWPerfectComplex_Ext0setGenerators):
    def __len__(self):
        raise NotImplementedError()

    def _coordinate_lift(self, quot_elem_in_coords):
        raise NotImplementedError()
        return quot_elem_in_coords.lift()
