from typing import Iterable, Any
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType

import operator

from sage.structure.parent import Parent

from sage.matrix.matrix0 import Matrix
from sage.matrix.constructor import matrix

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

from .gradings import (
    HomologicalGradingGroupElement,
    ExtendedQuiverGradingGroupElement,
)


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
        self._component_ranks = MappingProxyType(dict(component_ranks))
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

        result = "A graded free module with dimensions\n"
        result += pformat(dict(self._component_ranks))

        return result

    def is_instance_of_sameclass_or_shiftedclass(self, other):
        return isinstance(other, self.__class__ | self.ShiftedClass)

    def hom_set(self, other, **kwargs):
        if self.is_instance_of_sameclass_or_shiftedclass(other):
            return self.HomsetClass(domain=self, codomain=other, **kwargs)
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


class Graded_Homomorphism(ModuleElement):
    """
    Abstract class for graded homomorphisms.
    """

    __slots__ = ("_map_components",)

    def __init__(
        self,
        M: Parent,
        x: dict | Iterable,  # also a `Graded_Homomorphism`
        copy=True,
    ):
        ModuleElement.__init__(self, M)

        if isinstance(x, Graded_Homomorphism):
            _map_components = x._map_components
        elif isinstance(x, dict):
            _map_components = x
        elif isinstance(x, Iterable):
            _map_components = dict(x)
        else:
            raise ValueError(
                "Don't know how to make "
                + repr(x)
                + " into a homomorphism without copying."
            )

        if copy and not isinstance(x, Iterable):
            # already a copy if `x` is iterable
            import copy

            self._map_components = copy.copy(_map_components)
        else:
            self._map_components = _map_components

    def _add_(self, other):
        return self.parent()._element_constructor_(
            add(
                self._map_components,
                other._map_components,
            ),
            check=False,
            copy=False,
        )

    def _neg_(self):
        return self.parent()._element_constructor_(
            negate(self._map_components),
            check=False,
            copy=False,
        )

    def _sub_(self, other):
        return self.parent()._element_constructor_(
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
        grading = self.parent().hom_grading_group()(grading)
        if grading in self._map_components:
            if not sign:
                return self._map_components[grading]
            else:
                return self._negate_component_(self._map_components[grading])
        return None

    def __getitem__(self, shift):
        hom_shift = self.parent().hom_grading_group()(shift)
        shifted_components = {
            grading - hom_shift: mat for grading, mat in self._map_components.items()
        }
        return (
            self.parent()
            .shift(shift)
            ._element_constructor_(
                shifted_components,
                check=False,
                copy=False,
            )
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

    def set_immutable(self):
        for mat in self._map_components.values():
            mat.set_immutable()

    @staticmethod
    def _negate_component_(mat):
        """
        Return negative the component.

        In subclasses there is a faster way to do it.
        """
        return -mat


class GradedFreeModule_Homomorphism(Graded_Homomorphism):
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
        for deg, map in self.items():
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
        return self.parent()._element_constructor_(
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
        for deg, map in self.items():
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
        return self.parent()._element_constructor_(
            result_dict,
            check=False,
            copy=False,
        )

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
        **kwargs,
    ):
        return super().__classcall__(
            cls,
            domain=domain,
            codomain=codomain,
            base=base,
            **kwargs,
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
        return self.ring.opposite

    def _element_constructor_(
        self,
        x: Iterable | dict | GradedFreeModule_Homomorphism,
        check=True,
        ignore_checks: set[str] = set(),
        copy=True,
    ):
        if isinstance(x, Element):
            if x.parent() == self:
                return x

        if copy:
            _map_components = {}
            end_algebra = self.end_algebra
            if isinstance(x, dict):
                x = x.items()
            for deg, map in x:
                deg = self.hom_grading_group()(deg)
                if isinstance(map, Matrix):
                    map = map.change_ring(end_algebra)
                    _map_components[deg] = map
                    _map_components[deg].set_immutable()
                else:
                    _map_components[deg] = matrix(
                        end_algebra,
                        ncols=self.domain.component_rank(deg),
                        nrows=self.codomain.component_rank(deg),
                        entries=map,
                        sparse=True,
                        immutable=True,
                    )
        else:
            _map_components = x

        element = self.element_class(
            self,
            _map_components,
            copy=False,
        )

        # check conditions like commuting with differential
        # or having correct grading
        if check:
            # print("++++check")
            self._element_check_(element, ignore_checks)
        # else:
        #    print("----check")

        return element

    def hom_grading_group(self):
        return self.domain.hom_grading_group()

    def shift_group(self):
        return self.domain.shift_group()

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

    def _element_check_(self, element, ignore_checks: set[str] = set()):
        if "sizes" not in ignore_checks:
            # print("<0")
            for degree, mat in element:
                assert mat.ncols() == self.domain.component_rank(degree)
                assert mat.nrows() == self.codomain.component_rank(degree)

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
        `self.codomain.differential * morphism
        - morphism * self.domain.differential`;
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

        domain_diff = self.domain.differential
        codomain_diff = self.codomain.differential
        shifted_morphism = morphism[self.domain.differential.degree()]

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
            self.codomain[self.codomain.differential.degree()],
        )
        return cycle_module._element_constructor_(
            result_dict,
            check=False,
        )

    def differential(self):
        from sage.structure.unique_representation import unreduce

        # making differential's codomain by shifting
        # codomain in self by differential's degree.
        shift = self.codomain.differential.degree()
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


class Differential(GradedFreeModule_Homomorphism):
    @lazy_class_attribute
    def OriginalClass(cls):
        return cls

    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedDifferential

    def __init__(
        self,
        underlying_module: GradedFreeModule,
        differential_data: (
            dict[
                HomologicalGradingGroupElement | ExtendedQuiverGradingGroupElement,
                Matrix | dict,
            ]
            | Iterable[
                tuple[
                    HomologicalGradingGroupElement | ExtendedQuiverGradingGroupElement,
                    Matrix | dict,
                ]
            ]
            | GradedFreeModule_Homomorphism
        ),
        degree,
        grading_group=None,
        coerce_coefficients=True,
        check=True,
    ):
        if grading_group is None:
            self._shift_grading_group = underlying_module.shift_group()
        if isinstance(degree, int) or isinstance(degree, Integer):
            degree = underlying_module.hom_grading_group()(degree)
        self._degree = self._shift_grading_group(degree)
        assert self._degree.sign() == -1

        # We want to make differential into a morphism.
        # The correct homset is the one that does all the checks
        # except commuting with the differential
        # [even thought it is true when the differential
        # squares to zero, we can't check it before we
        # define self.parent().differential that uses this __init__]
        codomain = underlying_module[self._degree]
        graded_hom_set = underlying_module.hom_set(codomain, check=False)
        graded_map = graded_hom_set._element_constructor_(
            differential_data,
            check=check,
            ignore_checks=set(["chain"]),
        )
        super().__init__(
            graded_hom_set,
            graded_map,
            copy=False,
        )

        # now we check the square
        if check:
            diff_square = self * self
            assert diff_square.is_zero(), "Differential has non-zero square,\n" + repr(
                diff_square
            )

    def degree(self):
        return self._degree

    def hom_degree(self):
        return self.parent().hom_grading_group()(self._degree)

    def __repr__(self):
        result = "A differential of degree {} with components\n".format(self.degree())
        result += super().__repr__()

        return result

    def __getitem__(self, key):
        shift = self.parent().shift_group()(key)

        if shift:
            return self.ShiftedClass(
                self,
                shift=shift,
            )
        else:
            return self


@dataclass(frozen=True, eq=True, repr=False, slots=True)
class ShiftedDifferential(GradedFreeModule_Homomorphism):
    """
    Objects of this class are supposed to be created by [shift]
    operator of other modules, not by calling init directly.

    We do not inherit from ShiftedObject only because of
    issues with diamond inheritence with __slots__
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

    def __call__(self, grading=None, sign=False):
        sign = self.shift.sign()
        hom_shift = self.hom_shift()
        diff_component = self.original(
            grading + hom_shift,
            sign != 1,
        )
        return diff_component

    def support(self):
        hom_shift = self.hom_shift()
        return frozenset(grading - hom_shift for grading in self.original.support())

    def __iter__(self):
        hom_shift = self.hom_shift()
        sign = self.shift.sign()
        for grading in self.original.support:
            yield (
                grading - hom_shift,
                self.original(
                    grading + hom_shift,
                    sign != 1,
                ),
            )

    @cached_method
    def parent(self):
        return self.original.parent().shift(self.shift)

    def hom_shift(self):
        return self.original.parent().hom_grading_group()(self.shift)

    def degree(self):
        return self.original.degree()

    def hom_degree(self):
        return self.original.hom_degree()

    def __getitem__(self, key):
        extra_shift = self.original.parent().shift_group()(key)
        return self.shift_object(extra_shift)

    def _map_coefficients_(self):
        """
        Make a dict of map coefficients.

        We don't want to cache it for every singe shift
        to high avoid memory consumption.
        This is why we use a function + `__getattr__`
        instead of a lazy attribute.
        """
        return dict(iter(self))

    def __getattr__(self, name):
        if name == "_map_coefficients":
            return self._map_coefficients_()

    def set_immutable(self):
        self.original.set_immutable()


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

    @lazy_class_attribute
    def DifferentialClass(cls):
        return Differential

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
        differential = Graded_Homomorphism(
            None,
            differential,
            copy=False,
        )
        differential.set_immutable()

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
        self.differential = self.DifferentialClass(
            underlying_module=self,
            differential_data=differential,
            degree=differential_degree,
        )

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


@dataclass(frozen=True, eq=True, repr=False, slots=True)
class ShiftedComplexOfFreeModules(ShiftedGradedFreeModule):
    """
    A graded module with shifted gradings.
    """

    differential: ShiftedDifferential = field(init=False, compare=False)

    @lazy_class_attribute
    def OriginalClass(cls):
        return ComplexOfFreeModules

    def __post_init__(self):
        assert isinstance(self.original, ComplexOfFreeModules)

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        return cls.OriginalClass.HomsetClassWithoutDifferential

    def __getattr__(self, name):
        """
        Define an attribute `differential`.

        Since we use `frozen=True` we need to bypass the protection,
        we can't just use `@lazy_attribute`
        """
        if name == "differential":
            diff = self._differential()
            # bypass protection in frozen=True to change the entry
            # save the value of the differential
            object.__setattr__(
                self,
                "differential",
                diff,
            )
            return diff

    def _differential(self):
        return self.original.differential[self.shift]


class ComplexOfFreeModules_Homomorphism(GradedFreeModule_Homomorphism):
    def _cone_component_ranks(self):
        parent = self.parent()
        diff_degree = parent.domain.differential.degree()
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
        diff_degree = parent.domain.differential.degree()
        morphism_shifted = self[diff_degree]
        domain_shifted = parent.domain[diff_degree]
        codomain = parent.codomain

        gradings = frozenset(domain_shifted.differential.support())
        gradings |= frozenset(codomain.differential.support())
        gradings |= frozenset(morphism_shifted.support())

        diff_hom_degree = parent.domain.differential.hom_degree()
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
            differential_degree=self.parent().domain.differential.degree(),
            differential=self._cone_differential(),
        )


class ComplexOfFreeModules_Homset(GradedFreeModule_Homset):
    Element = ComplexOfFreeModules_Homomorphism

    @lazy_class_attribute
    def HomologyClass(cls):
        raise NotImplementedError()

    def __classcall__(
        cls,
        domain: GradedFreeModule | ShiftedGradedFreeModule,
        codomain: GradedFreeModule | ShiftedGradedFreeModule,
        base=None,
        check=False,
        **kwargs,
    ):
        if check:
            assert domain.differential.degree() == codomain.differential.degree()

        return super().__classcall__(
            cls,
            domain=domain,
            codomain=codomain,
            base=base,
            **kwargs,
        )

    def _element_check_(self, element, ignore_checks: set[str] = set()):
        if "chain" not in ignore_checks:
            # print("<2")
            image = self.apply_differential(element)
            assert image.is_zero(), "Morphism does not commute with differential."

        super()._element_check_(element, ignore_checks)

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
