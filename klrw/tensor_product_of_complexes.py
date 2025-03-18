from typing import Iterable
from types import MappingProxyType
from itertools import product

from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod

from klrw.free_complex import (
    GradedFreeModule,
    ShiftedGradedFreeModule,
    GradedFreeModule_Homomorphism,
    GradedFreeModule_Homset,
)
from klrw.multicomplex import (
    MulticomplexOfFreeModules,
    KLRWPerfectMulticomplex,
)
from klrw.perfect_complex import KLRWDirectSumOfProjectives


class HomomorphismTensorProduct(UniqueRepresentation):
    """
    Tensor product of homomorphisms over the default base.

    This is a singleton.
    Classcall to several homomomorphisms gives their tensor product.

    Domain and codomain have to implement
    a class method `tensor_product`.
    If the `ring()` of the tensor product domain/codomain
    has a method `tensor_product_of_elements`,
    then uses this method to multiply matrix entries.
    Otherwise, multiplies matrix entres.

    The tensor product object is graded by one-dimensional grading.
    Use `totalize=False` to get grading by `direct_sum`
    of the grading groups.
    """

    @staticmethod
    def __classcall__(cls, *homs, totalize=True):
        assert all(isinstance(hom, GradedFreeModule_Homomorphism) for hom in homs)
        assert homs, "Need at least one homomorphism"
        instance = cls.instance()
        return instance(*homs, totalize=totalize)

    def __init__(self):
        pass

    def __call__(self, *homs, totalize=True):
        domain = self._make_domain(*(hom.parent().domain for hom in homs))
        codomain = self._make_codomain(*(hom.parent().codomain for hom in homs))

        assert domain.ring() == codomain.ring()
        ring = domain.ring()
        tensor_product_method = self._define_product_on_matrix_elements(ring)
        it = product(*(iter(gr_map) for gr_map in homs))
        map_dict = {}
        for (*map_items,) in list(it):
            hom_degrees, map_components = zip(*map_items)
            hom_deg = domain.hom_grading_group().cartesian_product_of_elements(
                *hom_degrees
            )
            map_dict[hom_deg] = self._tensor_product_of_sparse_matrices(
                *map_components,
                tensor_product_on_matrix_elements=tensor_product_method,
                tensor_product_entries_parent=ring,
            )

        tensor_product = domain.hom(codomain, map_dict)
        if totalize:
            from klrw.multicomplex import TotalizationOfGradedFreeModule

            tensor_product = TotalizationOfGradedFreeModule.totalize_morphism(
                tensor_product
            )

        return tensor_product

    @staticmethod
    def _instance(cls):
        """
        For technical reasons, we need a static version
        of method `instance()` for pickling/unpickling.
        """
        return UniqueRepresentation.__classcall__(cls)

    @classmethod
    def instance(cls):
        """
        Return the instance.
        """
        return cls._instance(cls)

    def __reduce__(self):
        """
        For pickling.
        """
        return self.__class__._instance, (self.__class__,)

    @staticmethod
    def _make_domain(*domains):
        return domains[0].tensor_product(*domains, totalize=False)

    @staticmethod
    def _make_codomain(*codomains):
        return codomains[0].tensor_product(*codomains, totalize=False)

    @classmethod
    def _define_product_on_matrix_elements(cls, ring):
        try:
            return ring.tensor_product_of_elements
        except AttributeError:

            def tensor_product_on_matrix_elements(*elements):
                return prod(elements)

            return cls._default_tensor_product_on_matrix_elements

    @staticmethod
    def _default_tensor_product_on_matrix_elements(*elements):
        return prod(elements)

    @staticmethod
    def _index_in_tensor_product(multiindex, dims):
        index = 0
        for i, d in zip(multiindex, dims):
            index *= d
            index += i
        return index

    @classmethod
    def _tensor_product_of_sparse_matrices(
        cls,
        *matrices,
        tensor_product_on_matrix_elements,
        tensor_product_entries_parent,
    ):
        row_dimensions = tuple(mat.nrows() for mat in matrices)
        col_dimensions = tuple(mat.ncols() for mat in matrices)
        it = product(*(mat.dict(copy=False).items() for mat in matrices))
        matrix_dict = {}
        for (*elements_data,) in it:
            indices, entries = zip(*elements_data)
            row_indices, col_indices = zip(*indices)
            i = cls._index_in_tensor_product(row_indices, row_dimensions)
            j = cls._index_in_tensor_product(col_indices, col_dimensions)
            # entries are in the opposite algebra
            # make them elements of the original KLRW algebra
            entries = list(map(lambda x: x.value, entries))
            entry = tensor_product_on_matrix_elements(*entries)
            matrix_dict[i, j] = entry

        return matrix(
            tensor_product_entries_parent.opposite,
            nrows=prod(row_dimensions),
            ncols=prod(col_dimensions),
            entries=matrix_dict,
            sparse=True,
        )


class TensorProductOfGradedFreeModules(GradedFreeModule):
    """
    Returns tensor product of graded free modules.

    The grading groups have to implement
    static method `direct_sum`.
    The result has to implement `cartesian_product_of_elements`.
    The tensor product object is graded
    by `direct_sum` of the grading groups.
    Use `.totalization()` to get a one-dimensional grading.
    If base rings have a method `tensor_product`, then
    uses this for the base ring of a product.
    Otherwise, makes sure the base rings are the same,
    and takes tensor product over it.
    """

    @staticmethod
    def __classcall__(
        cls,
        *modules,
        totalize=True,
    ):
        return UniqueRepresentation.__classcall__(
            cls,
            *modules,
        )

    def __init__(
        self,
        *modules: Iterable[GradedFreeModule],
    ):
        assert all(
            isinstance(modul, GradedFreeModule | ShiftedGradedFreeModule)
            for modul in modules
        )
        assert modules, "Need at least one group"
        self._parts = modules

    @lazy_attribute
    def _grading_group(self):
        grading_groups = [modul._grading_group for modul in self._parts]
        return grading_groups[0].direct_sum(*grading_groups)

    @lazy_attribute
    def _component_ranks(self):
        it = product(*(modul.component_rank_iter() for modul in self._parts))
        _component_ranks = {}
        for (*comp_rank_items,) in it:
            hom_degrees, comp_ranks = zip(*comp_rank_items)
            hom_deg = self.hom_grading_group().cartesian_product_of_elements(
                *hom_degrees
            )
            component_rank = sum(comp_ranks)
            _component_ranks[hom_deg] = component_rank
        return MappingProxyType(_component_ranks)

    @lazy_attribute
    def _ring(self):
        rings = [modul.ring() for modul in self._parts]
        try:
            _ring = rings[0].tensor_product(*rings)
        except AttributeError:
            try:
                from klrw.misc import get_from_all_and_assert_equality

                _ring = get_from_all_and_assert_equality(lambda x: x.ring())
            except AssertionError:
                raise AssertionError(
                    "Don't know what base assign to the tensor product"
                )
        return _ring


class TensorProductOfComplexesOfFreeModules(MulticomplexOfFreeModules):
    """
    Tensor product of complexes.

    By default, returns a multicomplex, with several
    differential coming from each component in the product.
    To get a complex with one differential, use `.totalization()`.
    """

    @lazy_attribute
    def _differentials(self):
        _differentials = [None for _ in range(len(self._parts))]
        ones = [part.hom_set(part).one() for part in self._parts]
        for i in range(len(self._parts)):
            differential = self._parts[i].differential
            new_differential = HomomorphismTensorProduct(
                *(
                    self._parts[i].differential if j == i
                    else ones[j]
                    for j in range(len(self._parts))
                ),
                totalize=False,
            )
            sign = self.hom_grading_group().sign_from_part(i, differential.sign)
            degree = self.shift_group().summand_embedding(i)(differential.degree())
            _differentials[i] = self.DifferentialClass(
                underlying_module=self,
                differential_data=new_differential,
                sign=sign,
                degree=degree,
            )
        self._check_differentials_(_differentials)
        return _differentials
        diff_degree = self._domain.differential.degree()
        morphism_shifted = self._morphism[diff_degree]

        gradings = frozenset(self._domain_shifted.differential.support())
        gradings |= frozenset(self._codomain.differential.support())
        gradings |= frozenset(morphism_shifted.support())

        diff_hom_degree = self._domain.differential.hom_degree()
        differential = {}
        for grading in gradings:
            next_grading = grading + diff_hom_degree
            # the matrix has block structure
            # columns are splitted into two categories by left_block_size
            left_block_size = self._domain_shifted.component_rank(grading)
            # the total number of columns
            new_domain_rk = left_block_size + self._codomain.component_rank(grading)
            # rows are splitted into two categories by top_block_size
            top_block_size = self._domain_shifted.component_rank(next_grading)
            # the total number of rows
            new_codomain_rk = top_block_size + self._codomain.component_rank(
                next_grading
            )

            differential_component = matrix(
                self._morphism.parent().end_algebra,
                ncols=new_domain_rk,
                nrows=new_codomain_rk,
                sparse=True,
            )

            # warning: set_block works too slow for sparse matrices
            # because it does not take into account that the matrices
            # are sparse, and that the initial block is zero.
            top_left_block = self._domain_shifted.differential(grading)
            # differential_component.set_block(0, 0, top_left_block)
            for (a, b), entry in top_left_block.dict(copy=False).items():
                differential_component[a, b] = entry
            bottom_left_block = morphism_shifted(grading)
            # differential_component.set_block(top_block_size, 0, bottom_left_block)
            for (a, b), entry in bottom_left_block.dict(copy=False).items():
                differential_component[top_block_size + a, b] = entry
            bottom_right_block = self._codomain.differential(grading)
            # differential_component.set_block(
            #     top_block_size, left_block_size, bottom_right_block
            # )
            for (a, b), entry in bottom_right_block.dict(copy=False).items():
                differential_component[top_block_size + a, left_block_size + b] = entry
            if self.keep_subdivisions:
                differential_component._subdivisions = (
                    [0, top_block_size, new_codomain_rk],
                    [0, left_block_size, new_domain_rk],
                )
            differential_component.set_immutable()
            differential[grading] = differential_component

        self.DifferentialClass(
            underlying_module=self,
            differential_data=differential,
            degree=diff_degree,
            sign=self._codomain.differential.sign,
            check=True,
        )


class TensorProductOfKLRWDirectSumsOfProjectives(
    TensorProductOfGradedFreeModules, KLRWDirectSumOfProjectives
):
    """
    Returns tensor product of direct sums of KLRW projectives.

    The shift groups have to implement static method
    `merge`, hom grading groups have to implement
    static method `direct_sum`. The result is graded
    by `direct_sum` of hom grading groups, and
    can be shifted by `merge` of shift groups.
    By default, returns a multicomplex, with several
    differential coming from each component in the product.
    Use `.totalization()` to get a one-dimensional hom grading
    with one differential.
    """

    @lazy_attribute
    def _KLRW_algebra(self):
        return self._ring

    @lazy_attribute
    def _extended_grading_group(self):
        grading_groups = [modul._extended_grading_group for modul in self._parts]
        return grading_groups[0].merge(*grading_groups)

    @lazy_attribute
    def _projectives(self):
        from klrw.perfect_complex import KLRWIrreducibleProjectiveModule

        _projectives = {}
        it = product(*(modul.projectives_iter() for modul in self._parts))
        for (*proj_items,) in it:
            hom_degrees, projs_tuples = zip(*proj_items)
            hom_deg = self.hom_grading_group().cartesian_product_of_elements(
                *hom_degrees
            )
            projectives_in_degree = []
            for projs in product(*projs_tuples):
                new_proj_list = []
                degree = self.KLRW_algebra().grading_group.zero()
                for pr in projs:
                    new_proj_list += pr.state.as_tuple()
                    degree += pr.equivariant_degree
                new_proj_state = self.KLRW_algebra().state(new_proj_list)
                new_proj = KLRWIrreducibleProjectiveModule(new_proj_state, degree)
                projectives_in_degree.append(new_proj)
            _projectives[hom_deg] = projectives_in_degree
        return MappingProxyType(_projectives)


class TensorProductOfKLRWPerfectComplexes(
    TensorProductOfKLRWDirectSumsOfProjectives,
    TensorProductOfComplexesOfFreeModules,
    KLRWPerfectMulticomplex,
):
    pass
