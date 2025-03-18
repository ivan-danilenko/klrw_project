from typing import Iterable
from collections import defaultdict
from types import MappingProxyType
from itertools import product, chain

from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.lazy_attribute import lazy_attribute
from sage.matrix.all import matrix

from klrw.free_complex import (
    ComplexOfFreeModules,
    ShiftedComplexOfFreeModules,
    ComplexOfFreeModules_Homomorphism,
)
from klrw.perfect_complex import KLRWPerfectComplex


class Cone(ComplexOfFreeModules):
    @staticmethod
    def __classcall__(
        cls,
        morphism: ComplexOfFreeModules_Homomorphism,
        keep_subdivisions=True,
    ):
        instance = UniqueRepresentation.__classcall__(
            cls,
            morphism=morphism,
        )
        instance.keep_subdivisions = keep_subdivisions
        return instance

    def __init__(
        self,
        morphism: ComplexOfFreeModules_Homomorphism,
    ):
        self._morphism = morphism
        parent = self._morphism.parent()
        self._domain = parent.domain
        diff_degree = self._domain.differential.degree()
        self._domain_shifted = self._domain[diff_degree]
        self._codomain = parent.codomain

    def morphism(self):
        return self._morphism

    def hom_grading_group(self):
        return self._domain.hom_grading_group()

    def shift_group(self):
        return self._domain.shift_group()

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


class KLRWCone(Cone, KLRWPerfectComplex):
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

    def lift_map(
        self,
        morphism,
        self_as_domain,
        allow_iterated_cones=True,
    ):
        """
        Constructs a map to or from `self`.

        Consider a pair of chain maps
        `f: A -> B`
        and
        `g: B -> C`.
        If `gf` is nil-homotopic, then we can construct maps
        `A[degree] -> cone(g)`
        and
        `cone(f) -> C`.
         -- if `self_as_domain=True`, then
            we construct `cone(f) -> C`
            for `g = morphism` (up to shift)
            and `cone(f) = self`.
        -- if `self_as_domain=False`, then
            we construct `A[degree] -> cone(g)`
            for `f = morphism` (up to shift)
            and `cone(g) = self`.

        Let's discuss why lifts exist and when they are unique.
        The exact triangle
        `B -> C -> cone(g) ->`
        gives rise to a long exact sequence of
        `Hom`s in the derived category
        `... -> Hom(A, C[-degree]) -> Hom(A, cone(g)[-degree]) ->`
        ` -> Hom(A, B) -> Hom(A, C) -> ...`;
        `gf` is nil-homotopic if and only if `f`
        is in the kernel of `Hom(A, B) -> Hom(A, C)`.
        This means that it's in the image of
        `Hom(A, cone(g)[-degree]) -> Hom(A, B)`, i.e.
        the lift exists.
        Lifts differ by chain maps `A -> C[-degree]`.
        Up to homotopy, this extra freedom is controlled
        by `Hom(A, C[-degree])`.
        We will be interested in the cases when
        `Hom(A, C[-degree]) = 0` and use the lift up to homotopy
        (for taking further cones).

        Similar for the case of `cone(f)` using
        `A -> B -> cone(f) ->`.

        Note: the cone over the lift `cone(f) -> C` is equal
        to the cone of the lift `A[degree] -> cone(g)` provided by
        the same graded map `A -> C[-degree]`.

        `morphism` can be shifted, the correct shift will
        be applied automatically.
        """
        from klrw.homotopy import homotopy
        from klrw.free_complex import ShiftedObject

        # We allow `morphism` to be a shifted map.
        # We first "unshift" it
        if self_as_domain:
            shift = ShiftedObject.find_shift(
                morphism.domain(),
                self._codomain,
            )
        else:
            shift = ShiftedObject.find_shift(
                morphism.codomain(),
                self._domain,
            )
        morphism = morphism[shift]

        if self_as_domain:
            composition = morphism * self._morphism
        else:
            composition = self._morphism * morphism

        diff_degree = self._domain.differential.degree()
        homotopy = homotopy(composition, verbose=False)
        if homotopy is None:
            raise ValueError("The product is not nil-homotopic.")
        # fix sign and degree
        homotopy = -homotopy[diff_degree]
        if self_as_domain:
            morphism_shifted = morphism
        else:
            morphism_shifted = morphism[diff_degree]

        new_morphism_dict = {}
        degrees = homotopy.support() | morphism_shifted.support()
        for hom_deg in degrees:
            homotopy_part = homotopy(hom_deg)
            morphism_part = morphism_shifted(hom_deg)

            if self_as_domain:
                map_component = homotopy_part.augment(morphism_part)
            else:
                map_component = morphism_part.stack(homotopy_part)

            new_morphism_dict[hom_deg] = map_component

        if self_as_domain:
            homset = self.hom_set(composition.codomain())
        else:
            homset = composition.domain()[diff_degree].hom_set(self)

        return homset(new_morphism_dict, check=False)


class KLRWIteratedCone(KLRWPerfectComplex):
    """
    Makes a complex from a sequence of morphisms.

    This is a generalized version of the construction
    in the "Note" of `KLRWCone.lift_map`
    (i.e. the cone over a lift of a map).
    Instead of starting with two composable morphisms,
    we take a sequece of morphisms
    `f_i: C_i -> C_{i+1}`
    forming a chain
    `C_0 -> C_1 -> ... -> C_n.`
    We form a differential of the iterated cone.
    Explicitly, it's lower block-triangular with
     -- Differentials of `C_i` on the diagonal,
     -- `f_i` on the subdiagonal,
     -- graded maps in other blocks; the condition
        on the differential to square to zero is
        equivalent to the requirement that these maps
        give a chain homotopy between zero and a map
        constructed from blocks closer to the diagonal.

    As a graded vector space, the output is a direct sum
    `C_0[n*diff_degree] + C_1[(n-1)*diff_degree] + ... + C_n`,
    where `diff_degree` is the degree of the differential.
    This is consistent with the gradings for the ordinary cone
    (i.e. the case of one morphism).

    Assumptions:
     -- Differentials of C_i square to zero,
     -- `f_i` are chain maps (i.e. commute with differentials),
     -- Compositions `f_{i+1}f_i` are nil-homotopic,
     -- `Hom(C_i, C_{i+k}[(1-k)*diff_degree]) = 0` for all `k>1`
        as `Hom`s in the homotopy category.
    The last two condition ensure that the homotopies that
    fill the lower blocks exist and are unique up to homotopy.
    This gives that the output exists and is unique
    up to isomorphism.

    Inputs:
     -- `morphisms` is an iterable of chain maps, `f_i`,
     -- `complexes` is an iterable of complexes `C_i`.
     -- `cache_level` controls how far from the diagonal
        caching is used. If it's `0`, no cache. If it's `1`,
        cache is applied to the first subdiagonal with homotopies.
    We allow some (or all) `C_i`s to be a direct sum of
    complexes. Then the corresponding term in `complexes`
    has to be an iterable of complexes, and `morphisms`
    to and from this term have to be a dictionary
    `{(i,j): chain_map_block}` representing a block matrix.
    """

    @staticmethod
    def __classcall__(
        cls,
        morphisms: Iterable[
            ComplexOfFreeModules_Homomorphism
            | dict[tuple[int, int], ComplexOfFreeModules_Homomorphism]
        ],
        complexes: Iterable[
            ComplexOfFreeModules
            | ShiftedComplexOfFreeModules
            | Iterable[ComplexOfFreeModules | ShiftedComplexOfFreeModules]
        ],
        cache_level=0,
    ):
        """
        We don't cache results via `UniqueRepresentation`.
        """
        from sage.misc.classcall_metaclass import typecall

        instance = typecall(cls, morphisms, complexes, cache_level)
        return instance

    def __init__(
        self,
        morphisms: Iterable[
            ComplexOfFreeModules_Homomorphism
            | dict[tuple[int, int], ComplexOfFreeModules_Homomorphism]
        ],
        complexes: Iterable[
            ComplexOfFreeModules
            | ShiftedComplexOfFreeModules
            | Iterable[ComplexOfFreeModules | ShiftedComplexOfFreeModules]
        ],
        cache_level=0,
    ):
        self._cache_level = cache_level
        self._counter = defaultdict(int)
        self._complexes = self._normalize_complexes_(complexes)
        self._morphisms = self._normalize_morphisms_(morphisms)
        assert len(self._complexes) == len(self._morphisms) + 1
        # the differential is split into blocks by splitting
        # `C_0[n*diff_degree] + C_1[(n-1)*diff_degree] + ... + C_n`
        # parts on the diagonal are (shifted);
        # each of these blocks, in turn, is also a (sparse)
        # block matrix because C_i's are direct sums.
        # We use indices `i, j, k` for the blocks (same for `C_i`s),
        # and `a, b, c` for the subblocks.
        self._off_diagonal_blocks = {}
        self._homotopy_cache = {}
        self._set_subdiagonal_()
        self._set_differential_degree_and_sign_()
        self._shift_off_diagonal_blocks_()
        self._complete_off_diagonal_part_()
        self._make_projectives_and_block_positions_()
        self._make_KLRW_algebra_()
        self._make_extended_grading_group_()
        self._make_differential_()
        self._del_auxilliary_data_()
        counter_of_counter = defaultdict(int)
        for x in self._counter.values():
            counter_of_counter[x] += 1
        print(dict(counter_of_counter))
        print("Done: ", sum(i * j for i, j in counter_of_counter.items()))
        print("Repeated: ", sum((i - 1) * j for i, j in counter_of_counter.items()))

    @staticmethod
    def _normalize_complexes_(complexes):
        """
        We make complexes a list of list
        and shift gradings.
        """
        complexes = tuple(complexes)
        length = len(complexes)
        normalized_complexes = [None for _ in range(length)]
        for i, complex_term in enumerate(complexes):
            if isinstance(
                complex_term, ComplexOfFreeModules | ShiftedComplexOfFreeModules
            ):
                complex_term = [complex_term]

            normalized_complexes[i] = [
                complex_[(length - i - 1) * complex_.differential.degree()]
                for complex_ in complex_term
            ]

        return normalized_complexes

    @staticmethod
    def _normalize_morphisms_(morphisms):
        morphisms = tuple(morphisms)
        normalized_morphisms = [None for _ in range(len(morphisms))]
        for i, morphism_component in enumerate(morphisms):
            if isinstance(morphism_component, ComplexOfFreeModules_Homomorphism):
                # if only one morphism is given
                # this is the only block
                morphism_component = {(0, 0): morphism_component}
            assert isinstance(morphism_component, dict)

            normalized_morphisms[i] = morphism_component

        return normalized_morphisms

    @staticmethod
    def _multiply_blocks_(
        first_block,
        second_block,
    ):
        from bisect import bisect_left, bisect_right

        if not first_block or not second_block:
            # if one of the blocks is empty, we get zero
            return {}

        result = {}

        # sort fist by column, then by rpw
        block_indices1 = sorted(first_block.keys(), key=lambda x: (x[1], x[0]))
        # sort fist by row, then by column
        block_indices2 = sorted(second_block.keys())

        # now we iterate over columns of the first matrix
        end_of_column_index1 = 0
        while end_of_column_index1 != len(block_indices1):
            start_of_column_index1 = end_of_column_index1
            # intermediate index in matrix product
            c = block_indices1[start_of_column_index1][1]
            # this will find the end of `c`-th column
            # by comparing the last coordinate
            end_of_column_index1 = bisect_right(
                block_indices1,
                c,
                lo=start_of_column_index1,
                key=lambda x: x[1],
            )

            # finding beginning and end of the k-th column
            # in the second block
            start_of_column_index2 = bisect_left(
                block_indices2,
                c,
                key=lambda x: x[0],
            )
            end_of_column_index2 = bisect_right(
                block_indices2,
                c,
                key=lambda x: x[0],
            )
            # if the column is empty, nothing to do
            if end_of_column_index2 == start_of_column_index2:
                continue

            it = product(
                range(start_of_column_index1, end_of_column_index1),
                range(start_of_column_index2, end_of_column_index2),
            )
            for inds in it:
                mat_indices1 = block_indices1[inds[0]]
                # the other coordinate is `c`, we ignore it
                a = mat_indices1[0]
                # same here
                mat_indices2 = block_indices2[inds[1]]
                b = mat_indices2[1]

                product_element = first_block[mat_indices1] * second_block[mat_indices2]

                if product_element.is_zero():
                    continue
                product_indices = a, b
                if product_indices in result:
                    result[product_indices] += product_element
                else:
                    result[product_indices] = product_element

        return result

    @staticmethod
    def _add_blocks_and_cleanup_(blocks):
        """
        We add in-place, so the first block will be modified.

        We also get rid of all zeroes.
        """
        blocks_it = iter(blocks)
        result = next(blocks_it)
        for bl in blocks_it:
            for indices, entry in bl.items():
                if indices in result:
                    result[indices] += entry
                else:
                    result[indices] = entry

        # clean-up
        result = {
            indices: entry for indices, entry in result.items() if not entry.is_zero()
        }

        return result

    def _set_subdiagonal_(self):
        for j in range(len(self._morphisms)):
            self._off_diagonal_blocks[j + 1, j] = self._morphisms[j]

    def _set_differential_degree_and_sign_(self):
        from klrw.misc import get_from_all_and_assert_equality

        complexes_iter = chain(*self._complexes)
        self._differential_degree, self._sign = get_from_all_and_assert_equality(
            lambda x: (x.differential.degree(), x.differential.sign), complexes_iter
        )

    def _shift_off_diagonal_blocks_(self):
        """
        Shifts off-diagonal blocks.

        The construction gives us that `(i, j)`-th
        block is a graded map `C_j -> C_i[(j-i+1)*diff_degree]`.
        We need it to be
        `C_j[(n-j)*diff_degree] -> C_j[(n-i)*diff_degree + diff_degree]`,
        where `n` is the number of morphisms (or the number of `C_i`s minus one).
        So we shift it by `(n-j)*diff_degree`.
        """
        length = len(self._complexes)
        for (_, j), subblock in self._off_diagonal_blocks.items():
            for (a, b), graded_map in subblock.items():
                subblock[a, b] = graded_map[
                    (length - 1 - j) * self._differential_degree
                ]

    def _complete_off_diagonal_part_(self):
        # order of filling in the blocks matter.
        # before (i, j) block can be filled it,
        # all blocks in i-th row to the rights
        # and all blocks in j-th column above
        # have to be filled in
        # `k` parametrizes subdiagonals.
        for k in range(2, len(self._complexes)):
            for j in range(len(self._complexes) - k):
                i = j + k
                print("Making ({},{}) block".format(i, j))
                self._complete_off_diagonal_block_(i, j)
                # print(len(self._off_diagonal_blocks[i, j]))

        column_width = max(
            (
                len(str(len(self._off_diagonal_blocks[i, j])))
                for j in range(len(self._complexes) - 2)
                for i in range(j + 1, len(self._complexes))
            )
        )

        # Print data rows
        for i in range(2, len(self._complexes)):
            print(
                " ".join(
                    f"{str(len(self._off_diagonal_blocks[i, j])):{column_width}}"
                    for j in range(i - 1)
                )
            )

    def _complete_off_diagonal_block_(self, i, j):
        #import multiprocessing as mp
        # when we compute the differential squared,
        # we get the condition
        # `differential(B_{i,j}) + B_{i,k}*B_{k,j} = 0'.
        # The summation is over `j < k < i`
        BikBkj = []
        for k in range(j + 1, i):
            Bik = self._off_diagonal_blocks[i, k]
            Bkj = self._off_diagonal_blocks[k, j]
            BikBkj_term = self._multiply_blocks_(Bik, Bkj)
            BikBkj.append(BikBkj_term)

        SumBikBkj = self._add_blocks_and_cleanup_(BikBkj)

        """
        mp_context = mp.get_context("spawn")
        with mp_context.Manager() as manager:
            tasks = [
                (
                    a,
                    b,
                    sum,
                    (i - j <= self._cache_level + 1),
                )
                for (a, b), sum in SumBikBkj.items()
            ]
            Bij = manager.dict()
            with mp_context.Pool(processes=8) as pool:
                homotopy_async_list = pool.starmap_async(
                    self._task_, tasks
                )
                Bij = dict(homotopy_async_list.get())
                print(Bij.keys())
        """
        Bij = {}
        for a, b in SumBikBkj:
            Bij[a, b] = -self._find_homotopy_(
                SumBikBkj[a, b],
                use_cache=(i - j <= self._cache_level + 1),
            )

        self._off_diagonal_blocks[i, j] = Bij

    @classmethod
    def _task_(cls, a, b, morphism, use_cache=False):
        return a, b, cls._task_find_homotopy_(morphism, use_cache=use_cache)

    @staticmethod
    def _task_find_homotopy_(morphism, use_cache=False):
        from klrw.homotopy import homotopy

        # If not in cache or cache not used.
        if morphism.domain().KLRW_algebra().base().invertible_parameters:
            max_iterations = 1
        else:
            max_iterations = 10
        homotopy_ = homotopy(
            morphism,
            verbose=False,
            max_iterations=max_iterations,
        )
        if homotopy_ is None:
            raise ValueError("Homotopy does not exist")
        if homotopy_.is_zero():
            return None

        return homotopy_

    def _find_homotopy_(self, morphism, use_cache=False):
        from klrw.homotopy import homotopy

        # First normalize and check if in cache.
        if use_cache:
            for _, mat in morphism:
                mat.set_immutable()

            from klrw.free_complex import ShiftedObject

            domain_original, domain_shift = ShiftedObject.original_and_shift(
                morphism.domain()
            )
            # We shift, and then shift back for more effective caching.
            morphism = morphism[-domain_shift]

            if morphism in self._homotopy_cache:
                homotopy_ = self._homotopy_cache[morphism]
                homotopy_ = homotopy_[domain_shift]
                # When we shift, we get signs for differential. Fix it.
                if domain_original.differential.sign(domain_shift) == -1:
                    homotopy_ = -homotopy_
                return homotopy_

        # If not in cache or cache not used.
        if morphism.domain().KLRW_algebra().base().invertible_parameters:
            max_iterations = 1
        else:
            max_iterations = 10
        homotopy_ = homotopy(
            morphism,
            verbose=False,
            max_iterations=max_iterations,
        )
        if homotopy_ is None:
            raise ValueError("Homotopy does not exist")
        if homotopy_.is_zero():
            return None

        # Cache if needed and shift homotopy back
        if use_cache:
            self._homotopy_cache[morphism] = homotopy_
            homotopy_ = homotopy_[domain_shift]
            # when we shift, we get signs for differential. Fix it.
            if domain_original.differential.sign(domain_shift) == -1:
                homotopy_ = -homotopy_
            self._counter[morphism.domain(), morphism.codomain()] += 1

        return homotopy_

    def _make_projectives_and_block_positions_(self):
        projectives = defaultdict(list)
        block_positions = {}
        for i, comp_list in enumerate(self._complexes):
            for a, comp in enumerate(comp_list):
                for grading, projs in comp.projectives_iter():
                    block_positions[grading, i, a] = len(projectives[grading])
                    projectives[grading] += projs

        self._projectives = MappingProxyType(projectives)
        self._block_positions = block_positions

    def _make_differential_(self):
        diff_hom_deg = self._differential_degree.homological_part()
        homological_degrees = (
            hom_deg
            for hom_deg in self._projectives
            if hom_deg + diff_hom_deg in self._projectives
        )

        differential_dict = {}
        for hom_deg in homological_degrees:
            differential_dict[hom_deg] = matrix(
                self.KLRW_algebra().opposite,
                ncols=self.component_rank(hom_deg),
                nrows=self.component_rank(hom_deg + diff_hom_deg),
                sparse=True,
            )

        # first fill in the diagonal
        for i, comp_list in enumerate(self._complexes):
            for a, comp in enumerate(comp_list):
                for hom_deg, mat in comp.differential:
                    # the top left corner of the block has coordinates
                    # (block_start, block_start)
                    block_row_start = self._block_positions[
                        hom_deg + diff_hom_deg, i, a
                    ]
                    block_col_start = self._block_positions[hom_deg, i, a]
                    diff_component = differential_dict[hom_deg]
                    for (b, c), entry in mat.dict(copy=False).items():
                        diff_component[b + block_row_start, c + block_col_start] = entry

        # now off-diagonal blocks
        for (i, j), block in self._off_diagonal_blocks.items():
            for (a, b), graded_map in block.items():
                for hom_deg, mat in graded_map:
                    block_row_start = self._block_positions[
                        hom_deg + diff_hom_deg, i, a
                    ]
                    block_col_start = self._block_positions[hom_deg, j, b]
                    diff_component = differential_dict[hom_deg]
                    for (c, d), entry in mat.dict(copy=False).items():
                        diff_component[c + block_row_start, d + block_col_start] = entry

        for mat in differential_dict.values():
            mat.set_immutable()

        self.differential = self.DifferentialClass(
            underlying_module=self,
            differential_data=differential_dict,
            degree=self._differential_degree,
            sign=self._sign,
        )

    def _del_auxilliary_data_(self):
        del self._homotopy_cache
        del self._complexes
        del self._morphisms
        del self._block_positions
        del self._off_diagonal_blocks
        del self._differential_degree
        del self._sign

    @lazy_attribute
    def _ring(self):
        return self._KLRW_algebra

    def _make_KLRW_algebra_(self):
        from klrw.misc import get_from_all_and_assert_equality

        complexes_iter = chain(*self._complexes)
        self._KLRW_algebra = get_from_all_and_assert_equality(
            lambda x: x.KLRW_algebra(), complexes_iter
        )

    def _make_extended_grading_group_(self):
        from klrw.misc import get_from_all_and_assert_equality

        complexes_iter = chain(*self._complexes)
        self._extended_grading_group = get_from_all_and_assert_equality(
            lambda x: x.shift_group(), complexes_iter
        )

    def __reduce__(self):
        """
        For pickling.
        """
        from functools import partial

        # make immutable
        projectives = {
            hom_deg: tuple(projs) for hom_deg, projs in self.projectives().items()
        }
        # make hashable
        projectives = frozenset(projectives.items())
        differential_data = frozenset(self.differential.items())

        return (
            partial(
                KLRWPerfectComplex,
                ring=self.ring(),
                projectives=projectives,
                differential=differential_data,
                differential_degree=self.differential.degree(),
                sign=self.differential.sign,
                extended_grading_group=self.shift_group(),
            ),
            (),
        )
