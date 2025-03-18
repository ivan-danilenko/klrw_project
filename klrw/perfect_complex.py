from typing import Iterable, Any
from collections import defaultdict
from dataclasses import dataclass, InitVar
from types import MappingProxyType
from itertools import product

import operator

from sage.matrix.constructor import matrix

from sage.modules.free_module import FreeModule

from sage.rings.integer_ring import ZZ
from sage.rings.ring import Ring
from sage.categories.action import Action

from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute, lazy_class_attribute
from sage.structure.unique_representation import UniqueRepresentation

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
from .free_complex import (
    GradedFreeModule,
    ShiftedGradedFreeModule,
    GradedFreeModule_Homomorphism,
    GradedFreeModule_Homset,
    ComplexOfFreeModules,
    ShiftedComplexOfFreeModules,
    ComplexOfFreeModules_Homomorphism,
    ComplexOfFreeModules_Homset,
    Differential,
    ShiftedDifferential,
    HomHomMultiplication,
)


@dataclass(frozen=True, init=True, eq=True, repr=False)
class KLRWIrreducibleProjectiveModule:
    state: KLRWstate
    equivariant_degree: QuiverGradingGroupElement | int = 0
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


class CenterHomMultiplication(Action):
    """
    Multiplication of a hom of graded projectives
    by a symmetric homogeneous element in the dot algebra.

    Default coercion uses the unit in KLRW which is not the best
    way to do multiplication by parameters.
    """

    def __init__(self, other, hom_set):
        self.dot_algebra = hom_set.KLRW_algebra().base()
        self.coerce_map = self.dot_algebra.coerce_map_from(other)
        Action.__init__(self, G=other, S=hom_set, is_left=True, op=operator.mul)

    def codomain(self):
        raise AttributeError("Codomain depend on the dot algebra element")

    def _act_(self, g, h, check=True):
        g = self.coerce_map(g)
        grading_group = self.domain().KLRW_algebra().grading_group
        element_degree = self.dot_algebra.element_degree(
            g,
            grading_group,
            check_if_homogeneous=True,
        )
        if check:
            assert self.dot_algebra.is_element_symmetric(
                g
            ), "The element is not symmetric"

        chain_map_dict = {}
        for hom_deg, map in h:
            chain_map_dict[hom_deg] = {
                (i, j): g * entry for (i, j), entry in map.dict(copy=False).items()
            }

        product_domain = self.domain().domain
        product_codomain = self.domain().codomain[0, element_degree]
        product_homset = product_domain.hom_set(product_codomain)
        chain_map = product_homset(chain_map_dict, check=False)
        return chain_map


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

    @lazy_class_attribute
    def TensorClass(cls):
        from klrw.tensor_product_of_complexes import (
            TensorProductOfKLRWDirectSumsOfProjectives,
        )

        return TensorProductOfKLRWDirectSumsOfProjectives

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
        homological_grading_names: Iterable[Any] | None = None,
        homological_grading_group: HomologicalGradingGroup | None = None,
        extended_grading_group: ExtendedQuiverGradingGroup | None = None,
        **kwargs,
    ):
        _extended_grading_group = cls._normalize_extended_grading_group_(
            ring,
            homological_grading_names,
            homological_grading_group,
            extended_grading_group,
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
                grading = _extended_grading_group(grading)
                grading_hom_part = grading.homological_part()
                grading_eq_part = grading.equivariant_part()
            else:
                try:
                    if isinstance(grading, int):
                        grading = ZZ(grading)
                    grading = _extended_grading_group.homological_part(grading)
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
            homological_grading_group=homological_grading_group,
            extended_grading_group=extended_grading_group,
            **kwargs,
        )

    def __init__(
        self,
        ring: KLRWAlgebra,
        projectives: frozenset[
            tuple[HomologicalGradingGroupElement, KLRWIrreducibleProjectiveModule]
        ],
        homological_grading_names=None,
        homological_grading_group=None,
        extended_grading_group=None,
    ):
        self._KLRW_algebra = ring
        self._projectives = MappingProxyType(dict(projectives))
        self._extended_grading_group = self._normalize_extended_grading_group_(
            ring,
            homological_grading_names,
            homological_grading_group,
            extended_grading_group,
        )

    @staticmethod
    def _normalize_extended_grading_group_(
        ring,
        homological_grading_names=None,
        homological_grading_group=None,
        extended_grading_group=None,
    ):
        # count how many pieces of data are given
        grading_data_pieces = sum(
            1
            for x in (
                homological_grading_names,
                homological_grading_group,
                extended_grading_group,
            )
            if x is not None
        )
        # there has to be exactly one piece of data
        assert grading_data_pieces <= 1, "Too much data for grading is given"
        if extended_grading_group is None:
            if homological_grading_group is None:
                if homological_grading_names is None:
                    homological_grading_names = (None,)
                homological_grading_names = tuple(homological_grading_names)
                homological_grading_group = HomologicalGradingGroup(
                    homological_grading_names=homological_grading_names
                )
            extended_grading_group = ExtendedQuiverGradingGroup(
                equivariant_grading_group=ring.grading_group,
                homological_grading_group=homological_grading_group,
            )
        else:
            assert extended_grading_group.equivariant_part == ring.grading_group

        return extended_grading_group

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
    def _negate_component_(mat):
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
            grading: self(grading, sign=True) for grading in self.support()
        }

        return self.parent()._element_constructor_(
            neg_map_components,
            check=False,
            copy=False,
        )


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

    def _element_check_(self, element, ignore_checks: set[str] = set()):
        if "projectives" not in ignore_checks:
            # print("<1")
            isomorphism_to_algebra = self.end_algebra.isomorphism_to_algebra
            for degree, mat in element:
                domain_projs = self.domain.projectives(degree)
                codomain_projs = self.codomain.projectives(degree)
                for (i, j), elem in mat.dict(copy=False).items():
                    elem = isomorphism_to_algebra(elem)
                    codomain_pr = codomain_projs[i]
                    domain_pr = domain_projs[j]

                    right_state = elem.right_state(
                        check_if_all_have_same_right_state=True
                    )
                    assert codomain_pr.state == right_state, repr(codomain_pr.state)
                    left_state = elem.left_state(check_if_all_have_same_left_state=True)
                    assert domain_pr.state == left_state, repr(domain_pr.state)
                    eq_degree = elem.degree(check_if_homogeneous=True)
                    assert (
                        codomain_pr.equivariant_degree - domain_pr.equivariant_degree
                        == eq_degree
                    ), (
                        repr(codomain_pr.equivariant_degree)
                        + " "
                        + repr(domain_pr.equivariant_degree)
                        + " "
                        + repr(eq_degree)
                    )

        super()._element_check_(element, ignore_checks)

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

    def _get_action_(self, other, op, self_on_left):
        if op == operator.mul:
            if self.KLRW_algebra().base().has_coerce_map_from(other):
                return CenterHomMultiplication(other=other, hom_set=self)
        return super()._get_action_(other, op, self_on_left)

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
            self._homset.codomain.differential.degree()
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

                            # set_block doesn't take into account
                            # that matrices are sparse
                            for (a, b), entry in submatrix.dict(copy=False).items():
                                indices = (
                                    row_subdivision_index + a,
                                    column_subdivision_index + b,
                                )
                                diff_mat[indices] = entry
                            # diff_mat.set_block(
                            #     row_subdivision_index,
                            #     column_subdivision_index,
                            #     submatrix,
                            # )

        # matrix of right multiplication by d.
        # there is a sign -1
        diff_deg = self._homset.domain.differential.degree().homological_part()
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
            -self._homset.codomain.differential.degree()
        )
        return previous_homset.basis._differential_matrix_(keep_subdivisions)

    def __repr__(self):
        result = "A basis in "
        result += repr(self._homset)

        return result


class KLRWPerfectComplex(ComplexOfFreeModules, KLRWDirectSumOfProjectives):
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedKLRWPerfectComplex

    @lazy_class_attribute
    def HomsetClass(cls):
        return KLRWPerfectComplex_Homset

    @lazy_class_attribute
    def TensorClass(cls):
        from klrw.tensor_product_of_complexes import TensorProductOfKLRWPerfectComplexes

        return TensorProductOfKLRWPerfectComplexes

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        assert issubclass(cls.HomsetClass, KLRWDirectSumOfProjectives_Homset)
        return KLRWDirectSumOfProjectives_Homset

    @lazy_class_attribute
    def DifferentialClass(cls):
        return KLRWDifferential

    @staticmethod
    def sum(*complexes, keep_subdivisions=True):
        if len(complexes) == 1:
            return complexes[0]
        else:
            from klrw.sum_of_complexes import SumOfKLRWPerfectComplexes

            return SumOfKLRWPerfectComplexes(*complexes)

    def base_change(self, other: Ring):
        assert other.has_coerce_map_from(self.ring())
        return KLRWPerfectComplex(
            ring=other,
            differential=self.differential,
            sign=self.differential.sign,
            differential_degree=self.differential.degree(),
            projectives=self.projectives(),
            homological_grading_names=self.hom_grading_group().names(),
        )

    def __repr__(self):
        from pprint import pformat

        result = "A complex of projective modules\n"
        result += pformat(dict(self.projectives()))
        result += "\nwith differential\n"
        for grading in sorted(self.differential.support()):
            result += repr(grading) + " -> "
            result += repr(grading + self.differential.hom_degree())
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
    KLRWDirectSumOfProjectives_Homomorphism, ComplexOfFreeModules_Homomorphism
):
    # def _cone_projectives(self):
    #    parent = self.parent()
    #    diff_degree = parent.domain.differential.degree()
    #    domain_shifted = parent.domain[diff_degree]
    #    codomain = parent.codomain
    #
    #    projectives = defaultdict(list)
    #    for grading, projs in domain_shifted.projectives_iter():
    #        projectives[grading] += projs
    #    for grading, projs in codomain.projectives_iter():
    #        projectives[grading] += projs
    #
    #    return MappingProxyType(projectives)

    def cone(self, keep_subdivisions=True):
        from klrw.cones import KLRWCone

        return KLRWCone(self, keep_subdivisions)
        # return KLRWPerfectComplex(
        #    projectives=self._cone_projectives(),
        #    ring=self.parent().ring,
        #    differential_degree=self.parent().domain.differential.degree(),
        #    differential=self._cone_differential(),
        # )

    def homology_class(self):
        return self.parent().homology()(self)


class KLRWDifferential(KLRWPerfectComplex_Homomorphism, Differential):
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedKLRWDifferential


class ShiftedKLRWDifferential(KLRWPerfectComplex_Homomorphism, ShiftedDifferential):
    @lazy_class_attribute
    def OriginalClass(cls):
        return KLRWDifferential

    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedKLRWDifferential


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


class KLRWPerfectComplex_Ext0(KLRWPerfectComplex_Homomorphism):
    def representative(self):
        return self.parent().representative(self)

    def in_coordinates(self, check=False):
        return self.parent().in_coordinates(self, check=check)

    def _richcmp_(self, right, op):
        from sage.structure.richcmp import richcmp, op_EQ, op_NE

        if op == op_EQ or op == op_NE:
            op_is_eq = op == op_EQ
            if self.parent() != right.parent():
                return not op_is_eq
            if (self - right).is_zero():
                return op_is_eq
            return not op_is_eq

        raise NotImplementedError("Needs more tests.")
        return richcmp(
            self.in_coordinates(check=False), right.in_coordinates(check=False), op
        )

    def is_zero(self):
        # first check if zero without homotopy
        if self.representative().is_zero():
            return True
        return self.parent().is_nilhomotopic(self)

    def __hash__(self):
        return hash(self.in_coordinates())

    def cone(self):
        return self.representative().cone()


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
        cycle_set = self.CycleClass(
            domain=self.domain,
            codomain=self.codomain,
        )
        one_as_cycle = cycle_set.one()
        return self.coerce(one_as_cycle)

    def zero(self):
        cycle_set = self.CycleClass(
            domain=self.domain,
            codomain=self.codomain,
        )
        zero_as_cycle = cycle_set.zero()
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

    def in_coordinates(self, element, check=False):
        return self.basis._vector_from_element_(element, check=check)

    def is_nilhomotopic(self, element):
        boundary_module = self.basis._coordinate_boundary_module
        ambient_homset = self.basis._ambient_homset
        # in coordinates of the ambient set of cycles
        elem_in_coord = ambient_homset.basis._vector_from_element_(element)
        return elem_in_coord in boundary_module

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

    @lazy_attribute
    def _ambient_homset(self):
        return self._cycleset.basis._ambient_homset

    @lazy_attribute
    def _cycleset(self):
        return self._extset.CycleClass(
            domain=self._extset.domain,
            codomain=self._extset.codomain,
        )

    @lazy_attribute
    def _coordinate_cycle_module(self):
        return self._cycleset.basis.coordinate_free_module

    @lazy_attribute
    def _coordinate_boundary_module(self):
        differential = self._cycleset.basis.previous_differential
        return differential.sparse_matrix().column_module()

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

    def _vector_from_element_(self, elem, check=False, immutable=True):
        rep = elem.representative()
        vec = self._cycleset.basis._vector_from_element_(rep, check=check)
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
