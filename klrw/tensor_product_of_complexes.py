from typing import Iterable
from collections import defaultdict
from types import MappingProxyType

from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.lazy_attribute import lazy_attribute
from sage.matrix.constructor import matrix

from klrw.free_complex import (
    GradedFreeModule,
    GradedFreeModule_Homset,
    ComplexOfFreeModules,
    ShiftedComplexOfFreeModules,
    ComplexOfFreeModules_Homomorphism,
)
from klrw.multicomplex import (
    MulticomplexOfFreeModules,
    KLRWPerfectMulticomplex,
)
from klrw.perfect_complex import KLRWDirectSumOfProjectives


class TensorProductOfGradedFreeModules_Homset(GradedFreeModule_Homset):
    """
    Tensor product over the default base.

    If bases implement static method `tensor_product`,
    that returns an object implementing a method
    `tensor_product_of_elements`,
    then uses this object as a base. Otherwise, makes sure
    bases are the same, and takes the tensor product over
    the base ring.
    """

    def __init__(self, *hom_sets):
        base_rings = [hom.base().algebra for hom in hom_sets]
        try:
            product_ring = base_rings[0].tensor_product(*base_rings)
            tensor_product_on_matrix_elements = product_ring.tensor_product_of_elements
        except AttributeError:
            try:
                from klrw.misc import get_from_all_and_assert_equality
                from sage.misc.misc_c import prod

                product_ring = get_from_all_and_assert_equality(
                    lambda x: x, base_rings
                )
                tensor_product_on_matrix_elements = prod
            except AssertionError:
                raise AssertionError(
                    "Don't know what base assign to the tensor product"
                )

        self._product_ring = product_ring
        self._tensor_product_on_matrix_elements = tensor_product_on_matrix_elements

        domains = [hom.domain.algebra for hom in hom_sets]
        codomains = [hom.codomain.algebra for hom in hom_sets]

        assert hom_sets, "Need at least one dot algebra"
        assert all(isinstance(hom, GradedFreeModule_Homset) for hom in hom_sets)
        self._parts = hom_sets
        quiver = get_from_all_and_assert_equality(
            lambda x: x.quiver_data.quiver, dot_algebras
        )
        quiver_data = FramedDynkinDiagram_with_dimensions.with_zero_dimensions(quiver)
        # we make an array where `i`th entry is the quiver with
        # dimensions that are the sum of dimensions of the first `i`
        # dot algebras
        self._partial_quiver_data = [quiver_data.immutable_copy()]
        for alg in dot_algebras:
            for vertex, dim in alg.quiver_data.dimensions(copy=False).items():
                quiver_data[vertex] += dim
            self._partial_quiver_data.append(quiver_data.immutable_copy())

        # TODO: fix _reduction and get all the data from reduction?
        base_ring = get_from_all_and_assert_equality(
            lambda x: x.base_ring(), dot_algebras
        )
        order = get_from_all_and_assert_equality(lambda x: x.term_order(), dot_algebras)
        parameters_names = [
            "no_deformations",
            "default_vertex_parameter",
            "default_edge_parameter",
            "invertible_parameters",
        ]
        parameters = {}
        for name in parameters_names:
            parameters[name] = get_from_all_and_assert_equality(
                lambda x: getattr(x, name), dot_algebras
            )
        prefixes = get_from_all_and_assert_equality(lambda x: x.prefixes, dot_algebras)
        parameters |= prefixes

        super().__init__(
            domain=domain,
            codomain=codomain,
            base=base,
        )

    @cached_method
    def embedding(self, i):
        """
        Embeds a piece into the tensor product.

        Homsets are unital, so we can send `x` to
        `1 @ ... @ x @ ... @ 1`
        where x is on the `i`th position.
        """
        i = int(i)
        assert i >= 0
        assert i < len(self._parts)
        part_dot_algebra = self._parts[i]
        dims_before_part = self._partial_quiver_data[i]
        variables_images = [None] * part_dot_algebra.ngens()
        for index, var in part_dot_algebra.variables.items():
            if var.position is not None:
                if isinstance(index, DotVariableIndex):
                    vertex = index.vertex
                    new_index = DotVariableIndex(
                        vertex, index.number + dims_before_part[vertex]
                    )
                else:
                    new_index = index
                image = self.variables[new_index].monomial
                variables_images[var.position] = image

        return part_dot_algebra.hom(variables_images, codomain=self)

    def tensor_product_of_elements(self, *elements):
        """
        Construct an element from parts.
        """
        return prod(self.embedding(i)(element) for i, element in enumerate(elements))


class TensorProductOfGradedFreeModules(GradedFreeModule):
    """
    Returns tensor product of graded free modules.

    The shift groups have to implement static method
    `merge`, hom grading groups have to implement
    static method `direct_sum`. The result is graded
    by `direct_sum` of hom grading groups, and
    can be shifted by `merge` of shift groups.
    If `totalization=True`, then the result is
    totalized, and hom grading group is one-dimensional.
    """
    @staticmethod
    def __classcall__(
        cls,
        *modules,
        totalize=False,
    ):
        from klrw.free_complex import ShiftedObject
        originals, shifts = zip(
            *(
                ShiftedObject.original_and_shift(module)
                for module in modules
            )
        )
        tensor_product = UniqueRepresentation.__classcall__(
            cls,
            *originals,
        )
        prod_shift_group = tensor_product.shift_group()
        total_shift = prod_shift_group.cartesian_product_of_elements(*shifts)
        if totalize:
            tensor_product = tensor_product.totalize()
            total_shift = prod_shift_group.totalization_morphism(total_shift)
        return tensor_product[total_shift]

    def __init__(
        self,
        *originals: Iterable[GradedFreeModule],
    ):
        assert all(
            isinstance(orig, GradedFreeModule)
            for orig in originals
        )
        assert originals, "Need at least one group"
        self._parts = originals
        parent = self._morphism.parent()
        self._domain = parent.domain
        self._codomain = parent.codomain

        super().__init__(
            self,
            ring: Ring,
            component_ranks: frozenset[tuple[Element, int]],
            grading_group,
        )

    def hom_grading_group(self):
        raise NotImplementedError()

    def shift_group(self):
        raise NotImplementedError()

    def ring(self):
        return self._domain.ring()

    def component_rank(self, grading):
        return self._domain_shifted.component_rank(
            grading
        ) + self._codomain.component_rank(grading)

    def component_rank_iter(self):
        for grading, dim in self._domain_shifted.component_rank_iter():
            extra_dim = self._codomain.component_rank(grading)
            yield (grading, dim + extra_dim)
        domain_shifted_gradings = self._domain_shifted.gradings()
        for grading, dim in self._domain_shifted.component_rank_iter():
            if grading not in domain_shifted_gradings:
                yield (grading, dim)

    @lazy_attribute
    def differential(self):
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

        return self.DifferentialClass(
            underlying_module=self,
            differential_data=differential,
            degree=diff_degree,
            sign=self._codomain.differential.sign,
            check=True,
        )


class TensorProductOfComplexesOfFreeModules(MulticomplexOfFreeModules):
    pass


class TensorProductOfKLRWDirectSumsOfProjectives(
    KLRWDirectSumOfProjectives, TensorProductOfGradedFreeModules
):
    pass


class TensorProductOfKLRWPerfectComplexes(
    KLRWPerfectMulticomplex,
    TensorProductOfKLRWDirectSumsOfProjectives,
):
    @lazy_attribute
    def _KLRW_algebra(self):
        return self._morphism.parent().KLRW_algebra()

    @lazy_attribute
    def _extended_grading_group(self):
        return self._domain.shift_group()

    @lazy_attribute
    def _projectives(self):
        projectives = defaultdict(list)
        for grading, projs in self._domain_shifted.projectives_iter():
            projectives[grading] += projs
        for grading, projs in self._codomain.projectives_iter():
            projectives[grading] += projs

        return MappingProxyType(projectives)
