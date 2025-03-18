from collections import defaultdict
from itertools import product
from functools import cache
from types import MappingProxyType
from typing import Iterable, Generator
from copy import copy

from sage.matrix.constructor import matrix
from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute, lazy_class_attribute
from sage.combinat.root_system.cartan_type import CartanType_abstract
from sage.rings.integer_ring import ZZ
from sage.rings.ring import Ring
from sage.misc.misc_c import prod

from .framed_dynkin import (
    FramedDynkinDiagram_class,
    FramedDynkinDiagram_with_dimensions,
    NodeInFramedQuiver,
)
from .dot_algebra import DotVariableIndex
from .klrw_algebra import KLRWAlgebra
from .klrw_braid import KLRWbraid_set
from .klrw_state import KLRWstate
from .gradings import QuiverGradingHomologicalLabel, HomologicalGradingGroup
from .perfect_complex import (
    KLRWIrreducibleProjectiveModule,
    KLRWPerfectComplex,
)
from .perfect_complex_corrections import corrected_diffirential_csc
from .homotopy import chain_map_by_correction


class IndexManagmentModule(UniqueRepresentation):
    r"""
    Class to encapulate index operations in braided projectives.

    The complex looks like a product of `length`
    complexes concentrated in two homological degrees;
    two projectives in homological degree 0,
    and one projective in homological degree \pm 1.
    We use indices 0 and 2 for homological degree 0
    and 1 for homological degree 1.
    """

    def __init__(self, length: int):
        self.length = length

    all_indices = tuple(range(2))

    def state_by_index(
        self,
        multiindex: tuple,
        left_framing: NodeInFramedQuiver,
        sequence: Iterable[NodeInFramedQuiver],
        right_framing: NodeInFramedQuiver,
    ) -> tuple[NodeInFramedQuiver]:
        """
        Return the tuple of the state specified by the data.
        Each multiindex correspond to a projective in the complex.
        """
        state = [sequence[x] for x in range(len(sequence)) if multiindex[x] == 0]
        # after brading the right framing appears on the left
        state += [right_framing]
        state += [
            sequence[x] for x in range(len(sequence) - 1, -1, -1) if multiindex[x] == 1
        ]
        # after brading the left framing appears on the right
        state += [left_framing]
        state += [sequence[x] for x in range(len(sequence)) if multiindex[x] == 2]
        return tuple(state)

    @lazy_attribute
    def dimensions(self):
        pass

    @lazy_attribute
    def position_in_state(self, multiindex, index):
        """
        Return position in the state of the braid corresponding to index.
        """
        if multiindex[index] == 0:
            position = sum(1 for x in multiindex[:index] if x == 0)

        elif multiindex[index] == 1:
            position = sum(1 for x in multiindex[:index] if x == 0)
            position += sum(1 for x in multiindex[index + 1 :] if x != 2)
            # framing
            position += 1

        elif multiindex[index] == 2:
            position = index
            position += sum(1 for x in multiindex[index + 1 :] if x != 2)
            # framing
            position += 2

        else:
            raise ValueError(
                "Got an unexpected value in multiindex: {}".format(multiindex[index])
            )

        return position

    def _multiindex_iterator_(self) -> Generator[tuple, None, None]:
        r"""
        Generates all tuples of multiindices.

        The complex looks like a product of n
        complexes concentrated in two homological degrees;
        two projectives in homological degree 0,
        and one projective in homological degree \pm 1;
        it's -1 if braiding_sign = 1, and
        1 if braiding_sign = - 1.
        We use indices 0 and 2 for homological degree 0
        and 1 for homological degree \pm 1.
        The order refines the following partial order:
        (...,1,...) preceeds (...,0,...) and (...,2,...).
        This is imporant when we define differential maps
        in initialization of `StandardBraidedProjectives`.
        """
        return product(*([(1, 0, 2)] * self.length))

    @cached_method
    def _index_by_multiindex_(n: int) -> MappingProxyType[tuple, tuple[int, int]]:
        """
        Make a dictionary of indices from multiindices.

        Each multiindex gives a pair
        (homological degree, index within homological degree).
        Homological degree depends on braiding sign
        [for sign = -1 gradings are nonnegrative,
        for sign = 1 gradings are nonpositive]
        so instead we use
        (-sign*homological degree, index within homological degree),
        to have the first entry to be always positive.
        Here we construct a dictionary
        from multiindices of length n to such pairs.
        """
        index_by_multiindex = {}
        len_of_projectives = defaultdict(int)

        for multiindex in _multiindex_iterator_(n):
            homological_degree = sum(1 if ind == 1 else 0 for ind in multiindex)

            index_in_chain = len_of_projectives[homological_degree]
            index_by_multiindex[multiindex] = (homological_degree, index_in_chain)

            len_of_projectives[homological_degree] += 1

        return MappingProxyType(index_by_multiindex)

    @cached_method
    def _homological_degree_by_multiindex_(
        n: int,
    ) -> MappingProxyType[tuple, tuple[int, int]]:
        """
        Make a dictionary of indices from multiindices.

        Each multiindex gives a pair
        (homological degree, index within homological degree).
        Homological degree depends on braiding sign
        [for sign = -1 gradings are nonnegrative,
        for sign = 1 gradings are nonpositive]
        so instead we use
        (-sign*homological degree, index within homological degree),
        to have the first entry to be always positive.
        Here we construct a dictionary
        from multiindices of length n to such pairs.
        """
        index_by_multiindex = {}
        len_of_projectives = defaultdict(int)

        for multiindex in _multiindex_iterator_(n):
            homological_degree = sum(1 if ind == 1 else 0 for ind in multiindex)

            index_in_chain = len_of_projectives[homological_degree]
            index_by_multiindex[multiindex] = (homological_degree, index_in_chain)

            len_of_projectives[homological_degree] += 1

        return MappingProxyType(index_by_multiindex)


@cache
def state_by_index(multiindex, left_framing, sequence, right_framing):
    """
    The complex looks like a product of `len(sequence)`
    complexes concentrated in two homological degrees;
    two projectives in homological degree 0,
    and one projective in homological degree 1.
    We use indices 0 and 2 for homological degree 0
    and 1 for homological degree 1.
    """
    state = [sequence[x] for x in range(len(sequence)) if multiindex[x] == 0]
    # after brading the right framing appears on the left
    state += [right_framing]
    state += [
        sequence[x] for x in range(len(sequence) - 1, -1, -1) if multiindex[x] == 1
    ]
    # after brading the left framing appears on the right
    state += [left_framing]
    state += [sequence[x] for x in range(len(sequence)) if multiindex[x] == 2]
    return tuple(state)


def position_in_state(multiindex, index):
    """
    Return position in the state of the braid corresponding to index.
    """
    if multiindex[index] == 0:
        position = sum(1 for x in multiindex[:index] if x == 0)

    elif multiindex[index] == 1:
        position = sum(1 for x in multiindex[:index] if x == 0)
        position += sum(1 for x in multiindex[index + 1 :] if x != 2)
        # framing
        position += 1

    elif multiindex[index] == 2:
        position = index
        position += sum(1 for x in multiindex[index + 1 :] if x != 2)
        # framing
        position += 2

    else:
        raise ValueError(
            "Got an unexpected value in multiindex: {}".format(multiindex[index])
        )

    return position


def find_exts(
    domain,
    codomain,
    range_=None,
    max_eq_deg=10,
    min_eq_deg=-10,
    verbose=True,
):
    exts = {}
    ext_dims = {}
    d_h = QuiverGradingHomologicalLabel(None)

    if range_ is None:
        from itertools import product

        max_hom_deg = max(
            (-deg1 + deg2).coefficient(d_h)
            for deg1 in domain.gradings()
            for deg2 in codomain.gradings()
        )
        min_hom_deg = min(
            (-deg1 + deg2).coefficient(d_h)
            for deg1 in domain.gradings()
            for deg2 in codomain.gradings()
        )
        range_ = product(
            range(min_hom_deg, max_hom_deg + 1), range(min_eq_deg, max_eq_deg + 1)
        )

    for shift, i in range_:
        if verbose:
            print("Working on {} {}".format(shift, i))
        hom = domain.hom_set(codomain[shift, i])
        rhom = hom.homology()
        basis = rhom.basis
        dim = len(basis)
        if dim:
            exts[shift, i] = basis()
            ext_dims[shift, i] = dim

    return exts, ext_dims


def _multiindex_iterator_(n: int) -> Generator[tuple, None, None]:
    r"""
    Generates all tuples of multiindices.

    `n` - int; number of strands between the twisted punctures;
        equivalenty, it's the lenght of multiindices.

    The complex looks like a product of n
    complexes concentrated in two homological degrees;
    two projectives in homological degree 0,
    and one projective in homological degree \pm 1;
    it's -1 if braiding_sign = 1, and
    1 if braiding_sign = - 1.
    We use indices 0 and 2 for homological degree 0
    and 1 for homological degree \pm 1.
    The order refines the following partial order:
    (...,1,...) preceeds (...,0,...) and (...,2,...).
    This is imporant when we define differential maps
    in initialization of `StandardBraidedProjectives`.
    """
    return product(*([(1, 0, 2)] * int(n)))


@cache
def _index_by_multiindex_(n: int) -> MappingProxyType[tuple, tuple[int, int]]:
    """
    Make a dictionary of indices from multiindices.

    Each multiindex gives a pair
    (homological degree, index within homological degree).
    Homological degree depends on braiding sign
    [for sign = -1 gradings are nonnegrative,
    for sign = 1 gradings are nonpositive]
    so instead we use
    (-sign*homological degree, index within homological degree),
    to have the first entry to be always positive.
    Here we construct a dictionary
    from multiindices of length n to such pairs.
    """
    index_by_multiindex = {}
    len_of_projectives = defaultdict(int)

    for multiindex in _multiindex_iterator_(n):
        homological_degree = sum(1 if ind == 1 else 0 for ind in multiindex)

        index_in_chain = len_of_projectives[homological_degree]
        index_by_multiindex[multiindex] = (
            StandardBraidedProjectives.homological_grading_group(homological_degree),
            index_in_chain,
        )

        len_of_projectives[homological_degree] += 1

    return MappingProxyType(index_by_multiindex)


@cache
def _component_rank_(n: int) -> MappingProxyType[tuple, tuple[int, int]]:
    len_of_projectives = defaultdict(int)

    for multiindex in _multiindex_iterator_(n):
        homological_degree = sum(1 if ind == 1 else 0 for ind in multiindex)
        len_of_projectives[homological_degree] += 1

    return MappingProxyType(len_of_projectives)


class StandardBraidedProjectives(KLRWPerfectComplex):
    homological_grading_group = HomologicalGradingGroup(R=ZZ)
    ground_ring = ZZ
    klrw_options = {
        "vertex_scaling": True,
        "edge_scaling": True,
        "dot_scaling": True,
        "invertible_parameters": False,
    }

    @lazy_class_attribute
    def differential_degree(cls):
        return cls.homological_grading_group(-1)

    # @weak_cached_function(cache=128)  # automatically a staticmethod
    @staticmethod
    def __classcall__(
        cls,
        quiver: FramedDynkinDiagram_class | CartanType_abstract,
        left_framing: NodeInFramedQuiver,
        right_framing: NodeInFramedQuiver,
        sequence: Iterable[NodeInFramedQuiver],
        braiding_sign: int = 1,
    ):
        return UniqueRepresentation.__classcall__(
            cls,
            quiver=quiver,
            left_framing=left_framing,
            right_framing=right_framing,
            sequence=tuple(sequence),
            braiding_sign=int(braiding_sign),
        )

    def __init__(
        self,
        quiver: FramedDynkinDiagram_class | CartanType_abstract,
        left_framing: NodeInFramedQuiver,
        right_framing: NodeInFramedQuiver,
        sequence: Iterable[NodeInFramedQuiver],
        braiding_sign: int,
    ):
        assert braiding_sign in (
            -1,
            1,
        ), "The braiding sign must be -1 or 1, not {}".format(braiding_sign)

        self.quiver = quiver
        self.left_framing = left_framing
        self.right_framing = right_framing
        self.sequence = sequence
        self.braiding_sign = braiding_sign

        # KLRW_algebra = self._KLRW_algebra

        # complex_data = self._make_differential_and_projectives_(
        #    KLRW_algebra,
        #    left_framing,
        #    right_framing,
        #    sequence,
        #    braiding_sign,
        # )
        # differential, projectives = complex_data

        # super().__init__(
        #    ring=KLRW_algebra,
        #    projectives=projectives,
        #    differential=differential,
        #    differential_degree=self.differential_degree,
        # )

    @lazy_attribute
    def _extended_grading_group(self):
        # only one homological grading,
        # the default one
        return self._normalize_extended_grading_group_(
            ring=self._KLRW_algebra,
        )

    @lazy_attribute
    def differential(self):
        self._make_differential_and_projectives_()

        return self.differential

    @lazy_attribute
    def _projectives(self):
        self._make_differential_and_projectives_()

        return self._projectives

    def framed_dynkin_with_dimensions(self):
        DD = FramedDynkinDiagram_with_dimensions.with_zero_dimensions(
            quiver=self.quiver
        )

        DD[self.left_framing] += 1
        DD[self.right_framing] += 1
        for v in self.sequence:
            DD[v] += 1

        return DD

    @lazy_attribute
    def _KLRW_algebra(self):
        return KLRWAlgebra(
            self.ground_ring,
            self.framed_dynkin_with_dimensions(),
            **self.klrw_options,
        )

    def component_rank(self, grading):
        gen_label = self.homological_grading_group.indices()[0]
        grading = grading.coefficient(gen_label)
        grading = int(grading)
        grading *= -self.braiding_sign
        component_ranks = _component_rank_(len(self.sequence))
        if grading in component_ranks:
            return component_ranks[grading]
        else:
            return 0

    def component_rank_iter(self):
        for grading, lens in _component_rank_(len(self.sequence)).items():
            new_grading = self.homological_grading_group(grading)
            new_grading *= -self.braiding_sign
            yield (new_grading, lens)

    def _make_differential_and_projectives_(
        self,
        # KLRW_algebra: KLRWAlgebra,
        # left_framing: NodeInFramedQuiver,
        # right_framing: NodeInFramedQuiver,
        # sequence: Iterable[NodeInFramedQuiver],
        # braiding_sign: int = 1,
        max_iterations_for_corrections=10,
    ):
        grading_group = self.KLRW_algebra().grading_group

        differential_dicts = {
            self.homological_grading_group(-self.braiding_sign * hom_deg): {}
            for hom_deg in range(len(self.sequence) + 1)
        }
        projectives = {
            self.homological_grading_group(-self.braiding_sign * hom_deg): []
            for hom_deg in range(len(self.sequence) + 1)
        }
        if self.braiding_sign == 1:
            del differential_dicts[self.homological_grading_group(-len(self.sequence))]
        else:
            del differential_dicts[self.homological_grading_group(0)]

        index_by_multiindex = _index_by_multiindex_(len(self.sequence))

        for multiindex in _multiindex_iterator_(len(self.sequence)):
            state_tuple = state_by_index(
                multiindex, self.left_framing, self.sequence, self.right_framing
            )
            state = self.KLRW_algebra().state(state_tuple)

            homological_degree, index_in_chain = index_by_multiindex[multiindex]
            homological_degree *= -self.braiding_sign

            # caution: after brading right_framing is on the left
            # and left_framing is on the right
            # gradings from log(1-y_i/z_j)
            equivariant_degree = sum(
                grading_group.crossing_grading(self.right_framing, self.sequence[pos])
                for pos in range(len(multiindex))
                if multiindex[pos] != 0
            )
            equivariant_degree += sum(
                grading_group.crossing_grading(self.left_framing, self.sequence[pos])
                for pos in range(len(multiindex))
                if multiindex[pos] != 2
            )
            # gradings from log(1-y_i/y_j)
            equivariant_degree += sum(
                grading_group.crossing_grading(self.sequence[pos1], self.sequence[pos2])
                for pos1 in range(len(multiindex))
                for pos2 in range(pos1 + 1, len(multiindex))
                if multiindex[pos1] > multiindex[pos2]
            )
            equivariant_degree += sum(
                grading_group.crossing_grading(self.sequence[pos1], self.sequence[pos2])
                for pos1 in range(len(multiindex))
                for pos2 in range(pos1 + 1, len(multiindex))
                if multiindex[pos1] == 1 and multiindex[pos2] == 1
            )
            # sign should depend direction of brading
            equivariant_degree *= self.braiding_sign

            # now we make matrix elements
            # if `brading_sign = 1`
            # we make all elements that map *from* this multiindex,
            # if `brading_sign = -1`
            # we make all elements that map *to* this multiindex.
            # In both cases the other multiindex
            # is obtained by replacing a 0 or a 2 with a 1.
            # Quantities related to the other end
            # of the matrix element are called other_.
            iter = (pos for pos in range(len(multiindex)) if multiindex[pos] != 1)
            for pos in iter:
                left_part = multiindex[:pos]
                right_part = multiindex[pos + 1 :]
                # strictly speaking, this is not the degree, terms with index 2
                # contribute 0, not 2. It's correct modulo 2, which is the only
                # thing we need.
                degree = sum(left_part)
                sign = 1 if degree % 2 == 0 else -1
                other_multiindex = left_part + (1,) + right_part
                other_hom_degree, other_index_in_chain = index_by_multiindex[
                    other_multiindex
                ]
                other_hom_degree *= -self.braiding_sign

                # Now we find the positions of the only moving strand
                # ...
                # The position of left is determined by how many strands are
                # preceeding the moving one. We have:
                # left_framing is preceeding:
                other_position = 1
                # all elements in left_part with value 0 are preceeding:
                other_position += sum(1 for x in left_part if x == 0)
                # all elements in right_part with values 0 or 1 are preceeding:
                other_position += sum(1 for x in right_part if x != 2)

                if multiindex[pos] == 0:
                    # if this strand moves 1<->0
                    # Only elements in left_part with value 0 are preceeding:
                    position = sum(1 for x in left_part if x == 0)
                elif multiindex[pos] == 2:
                    # if this strand moves 2<->0
                    # Only elements in right_part with value 2 are *not* preceeding.
                    # Don't forget about 2 strands for framing!
                    # and shift -1 because numbering starts from 0
                    position = len(self.sequence) + 2 - 1
                    position -= sum(1 for x in right_part if x == 2)
                else:
                    raise ValueError(
                        "Got an unexpected value in multiindex: {}".format(
                            multiindex[pos]
                        )
                    )

                if self.braiding_sign == 1:
                    assert (
                        homological_degree - other_hom_degree == 1
                    ), "Unexpected homological degree: {} not 1.".format(
                        other_hom_degree - homological_degree
                    )
                    right_position = other_position
                    left_position = position
                    other_state_tuple = state_by_index(
                        other_multiindex,
                        self.left_framing,
                        self.sequence,
                        self.right_framing,
                    )
                    right_state = self.KLRW_algebra().state(other_state_tuple)
                else:
                    assert (
                        other_hom_degree - homological_degree == 1
                    ), "Unexpected homological degree: {} not 1.".format(
                        homological_degree - other_hom_degree
                    )
                    right_position = position
                    left_position = other_position
                    right_state = state

                braid = (
                    self.KLRW_algebra()
                    .braid_set()
                    .braid_for_one_strand(
                        right_state=right_state,
                        right_moving_strand_position=right_position,
                        left_moving_strand_position=left_position,
                    )
                )
                entry = self.KLRW_algebra().base()(sign) * self.KLRW_algebra().monomial(
                    braid
                )

                if self.braiding_sign == 1:
                    diff_dict = differential_dicts[homological_degree]
                    # diff_dict[index_in_chain, other_index_in_chain] = entry
                    diff_dict[other_index_in_chain, index_in_chain] = entry
                else:
                    diff_dict = differential_dicts[homological_degree + 1]
                    # diff_dict[other_index_in_chain, index_in_chain] = entry
                    diff_dict[index_in_chain, other_index_in_chain] = entry

            projectives[homological_degree].append(
                KLRWIrreducibleProjectiveModule(state, equivariant_degree)
            )

        """
        if max_iterations_for_corrections == 0:
            # now making corrections
            triples = ()
            for i in range(len(self.sequence)):
                color = self.sequence[i]
                # look for the next strand with the same color
                try:
                    k = next(
                        x for x in range(i + 1, len(self.sequence)) if self.sequence[x] == color
                    )
                except StopIteration:
                    continue

                # add all intermediate strands if color is adjoint in the quiver
                triples += tuple(
                    (i, j, k)
                    for j in range(i + 1, k)
                    if self.KLRW_algebra().quiver[self.sequence[j], color] < 0
                )

            patterns = [
                ((1, 2, 2), (2, 1, 1), 1),
                ((0, 0, 1), (1, 1, 0), 1),
                ((0, 1, 2), (1, 1, 0), -1),
                ((0, 2, 2), (2, 1, 0), 1),
            ]
            # The other possible choice is
            # (1, 2, 2), (2, 1, 1), 1
            # (0, 0, 1), (1, 1, 0), 1
            # (0, 1, 2), (2, 1, 1), -1
            # (0, 0, 2), (2, 1, 0), -1
            # they differ by a chain homotopy coming from
            # the chain map with one term
            # (0, 1, 2) -> (2, 1, 0)

            all_indices = tuple(range(3))

            for i, j, k in triples:
                first_color = self.sequence[i]
                second_color = self.sequence[j]
                dot_algebra = self.KLRW_algebra().base()
                t_ij = dot_algebra.edge_variable(first_color, second_color).monomial
                r_i = dot_algebra.vertex_variable(first_color).monomial
                common_coeffient = r_i * t_ij

                iterator_pieces = [patterns]
                iterator_pieces += [all_indices] * (len(self.sequence) - 3)
                iterator = product(*iterator_pieces)

                for pattern, *common_indices in iterator:
                    print("++")
                    if self.braiding_sign == 1:
                        left_special_indices, right_special_indices, sign = pattern
                    else:
                        right_special_indices, left_special_indices, sign = pattern

                    print(i, j, k, common_indices)
                    first_part = tuple(common_indices[:i])
                    second_part = tuple(common_indices[i : j - 1])
                    third_part = tuple(common_indices[j - 1 : k - 2])
                    fourth_part = tuple(common_indices[k - 2 :])

                    left_multiindex = (
                        first_part
                        + (left_special_indices[0],)
                        + second_part
                        + (left_special_indices[1],)
                        + third_part
                        + (left_special_indices[2],)
                        + fourth_part
                    )
                    right_multiindex = (
                        first_part
                        + (right_special_indices[0],)
                        + second_part
                        + (right_special_indices[1],)
                        + third_part
                        + (right_special_indices[2],)
                        + fourth_part
                    )

                    hom_degree, left_index_in_chain = index_by_multiindex[
                        left_multiindex
                    ]
                    _, right_index_in_chain = index_by_multiindex[right_multiindex]
                    hom_degree *= -self.braiding_sign

                    print(hom_degree, ": ", left_multiindex, " <-> ", right_multiindex)

                    # print("hom_deg", hom_degree)

                    # the first and third strands swap
                    # the second stays
                    # TODO: explain more carefully
                    first_index = len(first_part)
                    # print("1:", first_index)
                    left_first_position = position_in_state(
                        left_multiindex, first_index
                    )
                    right_third_position = position_in_state(
                        right_multiindex, first_index
                    )

                    second_index = len(first_part) + len(second_part) + 1
                    left_second_position = position_in_state(
                        left_multiindex, second_index
                    )
                    right_second_position = position_in_state(
                        right_multiindex, second_index
                    )

                    third_index = (
                        len(first_part) + len(second_part) + len(third_part) + 2
                    )
                    left_third_position = position_in_state(
                        left_multiindex, third_index
                    )
                    right_first_position = position_in_state(
                        right_multiindex, third_index
                    )
                    first_word = KLRWbraid_set.word_for_one_strand(
                        right_moving_strand_position=right_first_position,
                        left_moving_strand_position=left_first_position,
                    )

                    # print("2:", left_first_position, right_first_position)
                    # print("1:", first_word)
                    second_word = KLRWbraid_set.word_for_one_strand(
                        right_moving_strand_position=right_second_position,
                        left_moving_strand_position=left_second_position,
                    )
                    third_word = KLRWbraid_set.word_for_one_strand(
                        right_moving_strand_position=right_third_position,
                        left_moving_strand_position=left_third_position,
                    )
                    # The strands do not intersect.
                    # So first the first one moves, then the second one.
                    word = first_word + second_word + third_word

                    right_state_tuple = state_by_index(
                        right_multiindex, self.left_framing, self.sequence, self.right_framing
                    )
                    right_state = self.KLRW_algebra().state(right_state_tuple)
                    braid = self.KLRW_algebra().braid(
                        state=right_state,
                        word=word,
                    )

                    # print(braid)

                    # sign = 1
                    degree = sum(first_part) % 2
                    if degree == 1:
                        sign *= -1

                    coeff = sign * common_coeffient

                    element = self.KLRW_algebra().term(braid, coeff)

                    diff_dict = differential_dicts[hom_degree]
                    diff_dict[right_index_in_chain, left_index_in_chain] = element
                    # if self.braiding_sign == 1:
                    #    diff_dict[left_index_in_chain, right_index_in_chain] = element
                    # else:
                    #    diff_dict[left_index_in_chain, right_index_in_chain] = element
        """

        print(
            "--",
            self.left_framing,
            self.right_framing,
            self.sequence,
            self.braiding_sign,
        )

        if self.needs_corrections:
            diff_csc = corrected_diffirential_csc(
                klrw_algebra=self.KLRW_algebra(),
                differential=differential_dicts,
                projectives=projectives,
                degree=self.differential_degree,  # =-1
                parallel_processes=1,
                max_iterations_for_corrections=max_iterations_for_corrections,
                verbose=False,
            )

            differential_dicts = {
                hom_deg: mat.dict() for hom_deg, mat in diff_csc.items()
            }

        differential = {
            hom_deg: matrix(
                self.KLRW_algebra().opposite,
                len(projectives[hom_deg + self.differential_degree]),
                len(projectives[hom_deg]),
                mat,
                sparse=True,
                immutable=True,
            )
            for hom_deg, mat in differential_dicts.items()
        }

        self._projectives = projectives
        self.differential = self.DifferentialClass(
            underlying_module=self,
            differential_data=differential,
            degree=self.differential_degree,
        )

        return differential, projectives

    @lazy_attribute
    def needs_corrections(self):
        if len(self.sequence) < 3:
            return False
        for i in range(1, len(self.sequence) - 1):
            mid_color = self.sequence[i]
            needs_correction = any(
                self.quiver[color, mid_color] < 0 for color in self.sequence[:i]
            )
            needs_correction &= any(
                self.quiver[color, mid_color] < 0 for color in self.sequence[i + 1 :]
            )
            if needs_correction:
                return True
        return False

    # @lazy_attribute
    # def _reduction(self):
    #    klrw_options = self.KLRW_algebra()._reduction[2].copy()
    #    del klrw_options["quiver_data"]
    #    self_kwrds = {
    #        "quiver": self.KLRW_algebra().quiver,
    #        "left_framing": self.left_framing,
    #        "right_framing": self.right_framing,
    #        "sequence": self.sequence,
    #        "braiding_sign": self.braiding_sign,
    #    }
    #    return (
    #        self.__class__,
    #        (),
    #        klrw_options | self_kwrds,
    #    )

    # def __reduce__(self):
    #    """
    #    Return the arguments that have been passed to
    #    :meth:`__new__<object.__new__>` to construct this object,
    #    as per the pickle protocol.
    #
    #    Overloading __reduce__ changes the behaviour of UniqueRepresentation.
    #    It no longer makes a _reduction attribute.
    #
    #    .. SEEALSO::
    #        :meth:`WithPicklingByInitArgs.__classcall__`,
    #        :mod:`~sage.structure.unique_representation`
    #    """
    #    from sage.structure.unique_representation import unreduce
    #
    #    return (unreduce, self._reduction)

    def _replace_(self, **replacements):
        """
        Make a similar parent with several adjustments.

        Compare to _replace of named tuples.
        """
        from sage.structure.unique_representation import unreduce

        cls, args, kwrds = self._reduction
        new_kwrds = kwrds | replacements
        return unreduce(cls, args, new_kwrds)

    @cached_method
    def _internal_braided_crossing_morphism_(
        self,
        index: int,
        self_on_right: bool = True,
    ):
        """

        ...
        `index = 1` crosses the first two moving strands between the punctures.
        """
        assert index > 0
        assert index < len(self.sequence)

        index_by_multiindex = _index_by_multiindex_(len(self.sequence))
        sequence = self.sequence
        other_sequence = sequence[: index - 1]
        other_sequence += (sequence[index], sequence[index - 1])
        other_sequence += sequence[index + 1 :]

        chain_map_dict = {
            hom_deg: matrix(
                self.KLRW_algebra().opposite,
                self.component_rank(hom_deg),
                self.component_rank(hom_deg),
                sparse=True,
            )
            for hom_deg in self.gradings()
        }

        all_indices = tuple(range(3))
        iterator_pieces = [all_indices] * (len(self.sequence) - 1)
        multi_index_iterator = product(*iterator_pieces)

        # now we make all entries between indices of form
        # `(..., a, a, ...) -> (..., a, a, ...)`
        # we call `a` a special index
        for *other_indices, special_index in multi_index_iterator:
            left_part = other_indices[: index - 1]
            right_part = other_indices[index - 1 :]
            multiindex = left_part + [special_index, special_index] + right_part
            multiindex = tuple(multiindex)

            # degree = sum(left_part)
            sign = 1 if special_index % 2 == 0 else -1
            sign = self.KLRW_algebra().base()(sign)

            homological_degree, index_in_chain = index_by_multiindex[multiindex]
            homological_degree *= -self.braiding_sign

            if self_on_right:
                right_sequence = self.sequence
            else:
                right_sequence = other_sequence
            right_state_tuple = state_by_index(
                multiindex, self.left_framing, right_sequence, self.right_framing
            )
            right_state = self.KLRW_algebra().state(right_state_tuple)

            # Now we find the positions of the only crossing.
            # First we take into account if left_framing and
            # right_framing are preceeding the crossing
            # also take into account that indices for crossings
            # start with 1.
            crossing_index = special_index + 1

            if special_index == 0:
                # if we cross 0<->0, then the only preceeding
                # moving strands are the strands in left_part
                # with value 0
                crossing_index += sum(1 for x in left_part if x == 0)
            elif special_index == 1:
                # if we cross 1<->1, then the only preceeding
                # moving strands are the strands in left_part
                # with value 0 and in right_part with value *not* 2.
                crossing_index += sum(1 for x in left_part if x == 0)
                crossing_index += sum(1 for x in right_part if x != 2)
            elif special_index == 2:
                # if this strand moves 2<->2, then the only
                # *not* preceeding moving strands are the strands
                # in right_part with value 2.
                # There are `len(sequence) - 2` other strands.
                crossing_index += (
                    len(sequence) - 2 - sum(1 for x in right_part if x == 2)
                )
            else:
                raise ValueError("Unknown index: {}".format(special_index))

            braid = self.KLRW_algebra().braid(
                state=right_state,
                word=(crossing_index,),
            )

            chain_term = chain_map_dict[homological_degree]
            element = self.KLRW_algebra().term(braid, sign)
            chain_term[index_in_chain, index_in_chain] = element

        # we need to add corrections of form
        # `(..., a, b, ...) -> (..., b, a, ...)`
        # and
        # `(..., b, a, ...) -> coeff*(..., a, b, ...)`
        # with
        # `a < b` for braiding_sign = +1 and
        # `a > b` for braiding_sign = -1
        # and `coeff` being exactly the same element
        # in the dot algebra as appears in double-crossing relation
        # [e.g. 2.10 and, as a special case, in the first relation
        # of 2.8 in  https://arxiv.org/pdf/1111.1431.pdf].
        # We call `(a, b)` special indices
        special_indices = ([0, 1], [0, 2], [1, 2])
        # != for boolean is xor
        if (self.braiding_sign == 1) != self_on_right:
            special_indices = tuple([b, a] for a, b in special_indices)

        multi_index_iterator = product(*iterator_pieces)
        for *other_indices, ind in multi_index_iterator:
            left_part = other_indices[: index - 1]
            right_part = other_indices[index - 1 :]
            multiindex = left_part + special_indices[ind] + right_part
            # other_multiindex = left_part + special_indices_sw[ind] + right_part
            other_multiindex = left_part + special_indices[ind][::-1] + right_part
            multiindex = tuple(multiindex)
            other_multiindex = tuple(other_multiindex)

            homological_degree, index_in_chain = index_by_multiindex[multiindex]
            homological_degree *= -self.braiding_sign
            _, other_index_in_chain = index_by_multiindex[other_multiindex]

            # First we make terms
            # `(..., a, b, ...) -> (..., b, a, ...)`
            right_state_tuple = state_by_index(
                multiindex, self.left_framing, self.sequence, self.right_framing
            )
            right_state = self.KLRW_algebra().state(right_state_tuple)

            chain_term = chain_map_dict[homological_degree]
            element = self.KLRW_algebra().idempotent(right_state)
            if self_on_right:
                # chain_term[other_index_in_chain, index_in_chain] = element
                chain_term[index_in_chain, other_index_in_chain] = element
            else:
                # chain_term[index_in_chain, other_index_in_chain] = element
                chain_term[other_index_in_chain, index_in_chain] = element

            # Now we make terms
            # `(..., b, a, ...) -> coeff*(..., a, b, ...)`
            first_color = self.sequence[index - 1]
            second_color = self.sequence[index]
            if first_color == second_color:
                # in this case `coeff = 0`
                continue

            dot_algebra = self.KLRW_algebra().base()
            quiver = self.KLRW_algebra().quiver
            d_ij = -quiver[first_color, second_color]
            first_index, second_index = special_indices[ind]
            # Now we compute `coeff`
            # This piece of code is almost identical to a part in
            # `KLRWAlgebra._right_action_of_s_iter_`
            t_ij = dot_algebra.edge_variable(first_color, second_color).monomial
            if d_ij > 0:
                d_ji = -quiver[second_color, first_color]
                t_ji = dot_algebra.edge_variable(second_color, first_color).monomial

                first_position = self._position_among_same_color_(
                    color=first_color,
                    index=first_index,
                    left_part=left_part,
                    right_part=right_part,
                )
                second_position = self._position_among_same_color_(
                    color=second_color,
                    index=second_index,
                    left_part=left_part,
                    right_part=right_part,
                )
                x_first = dot_algebra.dot_variable(
                    first_color,
                    first_position,
                ).monomial
                x_second = dot_algebra.dot_variable(
                    second_color,
                    second_position,
                ).monomial
                coeff = t_ij * x_first ** (d_ij) + t_ji * x_second ** (d_ji)
            else:
                coeff = t_ij

            right_state_tuple = state_by_index(
                multiindex, self.left_framing, other_sequence, self.right_framing
            )
            right_state = self.KLRW_algebra().state(right_state_tuple)
            element = coeff * self.KLRW_algebra().idempotent(right_state)
            if self_on_right:
                # chain_term[index_in_chain, other_index_in_chain] = element
                chain_term[other_index_in_chain, index_in_chain] = element
            else:
                # chain_term[other_index_in_chain, index_in_chain] = element
                chain_term[index_in_chain, other_index_in_chain] = element

        degree_shift = self.KLRW_algebra().grading_group.crossing_grading(
            self.sequence[index - 1],
            self.sequence[index],
        )
        other = self._replace_(
            sequence=other_sequence,
        )
        needs_corrections = self.needs_corrections or other.needs_corrections

        if self_on_right:
            domain = other
            codomain = self
        else:
            domain = self
            codomain = other

        codomain = codomain[0, degree_shift]

        if needs_corrections:
            chain_map = chain_map_by_correction(
                chain_map_dict, domain, codomain, verbose=False, check=True,
            )
            if chain_map is None:
                raise ValueError("Could not correct the chain map.")
        else:
            chain_map = domain.hom(codomain, chain_map_dict)
        return chain_map.homology_class()

    def _position_among_same_color_(self, color, index, left_part, right_part) -> int:
        # if no strands are preceeding, then the position is 1.
        position = 1
        if index == 0:
            # if the index is 0, then the only preceeding
            # moving strands are the strands in left_part
            # with value 0;
            # take into account only matching color
            position += sum(
                1 for x, c in zip(left_part, self.sequence) if c == color and x == 0
            )
        elif index == 1:
            # if the index is 1, then the only preceeding
            # moving strands are the strands in left_part
            # with value 0 and in right_part with value *not* 2;
            # take into account only matching color
            position += sum(
                1 for x, c in zip(left_part, self.sequence) if c == color and x == 0
            )
            index_shift = len(self.sequence) - len(right_part)
            shifted_sequence = self.sequence[index_shift:]
            position += sum(
                1 for x, c in zip(right_part, shifted_sequence) if c == color and x != 2
            )
        elif index == 2:
            # if the index is 2, then the only preceeding
            # moving strands are the strands in left_part
            # and in right_part with value *not* 2;
            # take into account only matching color
            position += sum(1 for c in self.sequence[: len(left_part)] if c == color)
            index_shift = len(self.sequence) - len(right_part)
            shifted_sequence = self.sequence[index_shift:]
            position += sum(
                1 for x, c in zip(right_part, shifted_sequence) if c == color and x != 2
            )

        return position

    @cached_method
    def _boundary_braided_crossing_morphism_(
        self,
        left_framing: bool = True,
        self_on_right: bool = True,
    ):
        index_by_multiindex = _index_by_multiindex_(len(self.sequence))
        if left_framing:
            moving_color = self.sequence[0]
            framing_color = self.left_framing
            other_sequence = self.sequence[1:]
        else:
            moving_color = self.sequence[-1]
            framing_color = self.right_framing
            other_sequence = self.sequence[:-1]
        other_projective_state_tuple = (self.left_framing,)
        other_projective_state_tuple += other_sequence
        other_projective_state_tuple += (self.right_framing,)
        if left_framing:
            other_projective_state_tuple = (
                moving_color,
            ) + other_projective_state_tuple
        else:
            other_projective_state_tuple = other_projective_state_tuple + (
                moving_color,
            )

        other_projective_state = self.KLRW_algebra().state(other_projective_state_tuple)
        state = other_projective_state
        klrw_options = self.KLRW_algebra()._reduction[2].copy()
        del klrw_options["quiver_data"]
        self_kwrds = {
            "quiver": self.KLRW_algebra().quiver,
            "state": state,
            "elementary_transposition": self.braiding_sign,
        }
        other = BraidedProjective(**(klrw_options | self_kwrds))

        other_index_by_multiindex = _index_by_multiindex_(len(self.sequence) - 1)

        if self_on_right:
            domain = other
            codomain = self
        else:
            domain = self
            codomain = other
        chain_map_dict = {
            hom_deg: matrix(
                self.KLRW_algebra().opposite,
                codomain.component_rank(hom_deg),
                domain.component_rank(hom_deg),
                sparse=True,
            )
            for hom_deg in self.gradings()
        }

        # != is xor for bools
        # or, equivalently, addition modulo 2.
        sub_or_quotient = self_on_right != (self.braiding_sign == 1)

        all_indices = tuple(range(3))
        iterator_pieces = [all_indices] * (len(self.sequence) - 1)
        multi_index_iterator = product(*iterator_pieces)

        # if sub_or_quotient:
        # ???now we make all entries between indices of form
        # ???`(0, ...) -> (...)`
        # ???`(..., 2) -> (...)`
        # ???or the other way around, depending on self_on_right
        for (*other_indices,) in multi_index_iterator:
            other_indices = tuple(other_indices)
            if left_framing:
                self_indices = (0,) + other_indices
            else:
                self_indices = other_indices + (2,)

            homological_degree, index_in_chain = index_by_multiindex[self_indices]
            homological_degree *= -self.braiding_sign
            _, other_index_in_chain = other_index_by_multiindex[other_indices]

            right_state_tuple = state_by_index(
                self_indices, self.left_framing, self.sequence, self.right_framing
            )
            right_state = self.KLRW_algebra().state(right_state_tuple)

            chain_term = chain_map_dict[homological_degree]
            idempotent = self.KLRW_algebra().idempotent(right_state)
            if sub_or_quotient:
                if self_on_right:
                    chain_term[index_in_chain, other_index_in_chain] = idempotent
                else:
                    chain_term[other_index_in_chain, index_in_chain] = idempotent
            else:
                if left_framing:
                    moving_position = 1
                else:
                    # last dot of its color
                    quiver_data = self.KLRW_algebra().base().quiver_data
                    moving_position = quiver_data[moving_color]
                # Now we compute `coeff`
                # This is the dot that appears in double-crossing relation
                # [e.g. 2.10 in https://arxiv.org/pdf/1111.1431.pdf].
                # It is a special case of the coefficient in
                # `KLRWAlgebra._right_action_of_s_iter_`
                t_ij = (
                    self.KLRW_algebra()
                    .base()
                    .edge_variable(moving_color, framing_color)
                    .monomial
                )
                # TODO: can simplify?
                # always 1 if `left_framing is True`
                # always last if `left_framing is False`
                coeff = t_ij
                if framing_color.node == moving_color.node:
                    x_dot = (
                        self.KLRW_algebra()
                        .base()
                        .dot_variable(
                            moving_color,
                            moving_position,
                        )
                        .monomial
                    )
                    coeff *= x_dot

                if self_on_right:
                    chain_term[index_in_chain, other_index_in_chain] = (
                        coeff * idempotent
                    )
                else:
                    chain_term[other_index_in_chain, index_in_chain] = (
                        coeff * idempotent
                    )

                if left_framing:
                    self_indices = (2,) + other_indices
                    other_extended_indices = (0,) + other_indices
                    self_moving_position = (
                        len(self.sequence) + 1 - sum(1 for x in other_indices if x == 2)
                    )
                    other_moving_position = 0
                else:
                    self_indices = other_indices + (0,)
                    other_extended_indices = other_indices + (2,)
                    self_moving_position = sum(1 for x in other_indices if x == 0)
                    other_moving_position = len(self.sequence) + 1

                homological_degree, index_in_chain = index_by_multiindex[self_indices]
                homological_degree *= -self.braiding_sign
                _, other_index_in_chain = other_index_by_multiindex[other_indices]

                if self_on_right:
                    right_position = self_moving_position
                    left_position = other_moving_position
                    self_state_tuple = state_by_index(
                        self_indices,
                        self.left_framing,
                        self.sequence,
                        self.right_framing,
                    )
                    right_state = self.KLRW_algebra().state(self_state_tuple)
                else:
                    right_position = other_moving_position
                    left_position = self_moving_position
                    other_state_tuple = state_by_index(
                        other_extended_indices,
                        self.left_framing,
                        self.sequence,
                        self.right_framing,
                    )
                    right_state = self.KLRW_algebra().state(other_state_tuple)

                braid = (
                    self.KLRW_algebra()
                    .braid_set()
                    .braid_for_one_strand(
                        right_state=right_state,
                        right_moving_strand_position=right_position,
                        left_moving_strand_position=left_position,
                    )
                )
                element = self.KLRW_algebra().term(
                    braid, self.KLRW_algebra().base()(-1)
                )

                chain_term = chain_map_dict[homological_degree]
                if self_on_right:
                    chain_term[index_in_chain, other_index_in_chain] = element
                else:
                    chain_term[other_index_in_chain, index_in_chain] = element

        if (
            len(self.sequence) >= 2
            and not sub_or_quotient
            and framing_color.node == moving_color.node
        ):
            if left_framing:
                extra_indices = (1, 2)
            else:
                extra_indices = (0, 1)

            relevant_positions = (
                pos
                for pos in range(len(other_sequence))
                if other_sequence[pos] == moving_color
            )

            t_ij = (
                self.KLRW_algebra()
                .base()
                .edge_variable(moving_color, framing_color)
                .monomial
            )
            r_i = self.KLRW_algebra().base().vertex_variable(moving_color).monomial
            coeff = t_ij * r_i
            if not left_framing:
                coeff *= -1

            # TODO: detailed description
            # the interaction of the first/last
            # [depending on `left_framing`]
            # and the corresponding framing may cause
            # corrections.
            # They are of form ???
            # `(b, a, ...) -> coeff*(a, b, ...)`
            # with
            # `a < b` for braiding_sign = +1 and
            # `a > b` for braiding_sign = -1
            # ???
            # and `coeff` being exactly the same element
            # in the dot algebra as appears in ReidemeisterIII-like
            # relation
            # [e.g. 2.14 in https://arxiv.org/pdf/1111.1431.pdf].
            # We call `(a, b)` special indices

            iterator_pieces = [relevant_positions]
            iterator_pieces += [all_indices] * (len(self.sequence) - 2)
            iterator_pieces += [extra_indices]
            multi_index_iterator = product(*iterator_pieces)
            for pos, *common_indices, ind in multi_index_iterator:
                left_part = tuple(common_indices[:pos])
                right_part = tuple(common_indices[pos:])

                if left_framing:
                    middle_part = left_part
                else:
                    middle_part = right_part
                # applying rule banning some combinations
                if ind != 1:
                    if 1 in middle_part:
                        continue

                # The other side has form
                # (..., ind, ...)
                # where ind = 1 or ind = 2 if `left_framing = True`
                # and ind = 0 or ind = 1 if `left_framing = False`

                other_indices = left_part + (ind,) + right_part
                if left_framing:
                    # One side has form
                    # (ind, ... , 0, ...)
                    # for the same ind as in other.
                    self_indices = (ind,) + left_part + (0,) + right_part
                    # positions of strands corresponding to 0:
                    self_first_position = sum(1 for x in left_part if x == 0)
                    # in other 0 is always on the first position
                    other_first_position = 0
                    # positions of strands corresponding to ind:
                    if ind == 1:
                        # in this case preceeding indices are everything with 0 or 1,
                        # one extra added 0, and left framing.
                        self_second_position = (
                            sum(1 for x in left_part if x != 2)
                            + sum(1 for x in right_part if x != 2)
                            + 2
                        )
                        # in other 1's in right_part do not contribute
                        other_second_position = self_second_position - sum(
                            1 for x in left_part if x == 1
                        )
                    else:  # ind == 2
                        # in this case following indices are everything with 2,
                        self_second_position = (
                            len(self.sequence)
                            + 1
                            - sum(1 for x in left_part if x == 2)
                            - sum(1 for x in right_part if x == 2)
                        )
                        # in other 2's in left_part do not contribute
                        other_second_position = (
                            len(self.sequence)
                            + 1
                            - sum(1 for x in right_part if x == 2)
                        )
                else:
                    # One side has form
                    # (... , 2, ..., ind)
                    # for the same ind as in other.
                    self_indices = left_part + (2,) + right_part + (ind,)
                    # positions of strands corresponding to 2:
                    self_second_position = (
                        len(self.sequence) + 1 - sum(1 for x in right_part if x == 2)
                    )
                    # in other 2 is always on the last position
                    other_second_position = len(self.sequence) + 1
                    # positions of strands corresponding to ind:
                    if ind == 1:
                        # in this case preceeding indices are everything with 0,
                        # and left framing.
                        self_first_position = (
                            sum(1 for x in left_part if x == 0)
                            + sum(1 for x in right_part if x == 0)
                            + 1
                        )
                        # in other 1's in right_part also contribute
                        other_first_position = self_first_position + sum(
                            1 for x in right_part if x == 1
                        )
                    else:  # ind == 0
                        # in this case following indices are everything with 2,
                        self_first_position = sum(1 for x in left_part if x == 0) + sum(
                            1 for x in right_part if x == 0
                        )
                        # in other 2's in right_part do not contribute
                        other_first_position = sum(1 for x in left_part if x == 0)

                if self_on_right:
                    first_word = KLRWbraid_set.word_for_one_strand(
                        right_moving_strand_position=self_first_position,
                        left_moving_strand_position=other_first_position,
                    )
                    second_word = KLRWbraid_set.word_for_one_strand(
                        right_moving_strand_position=self_second_position,
                        left_moving_strand_position=other_second_position,
                    )
                else:
                    first_word = KLRWbraid_set.word_for_one_strand(
                        right_moving_strand_position=other_first_position,
                        left_moving_strand_position=self_first_position,
                    )
                    second_word = KLRWbraid_set.word_for_one_strand(
                        right_moving_strand_position=other_second_position,
                        left_moving_strand_position=self_second_position,
                    )
                # The strands do not intersect.
                # So first the first one moves, then the second one.
                word = first_word + second_word

                hom_degree, index_in_chain = index_by_multiindex[self_indices]
                hom_degree *= -self.braiding_sign
                _, other_index_in_chain = other_index_by_multiindex[other_indices]

                if self_on_right:
                    right_state_tuple = state_by_index(
                        self_indices,
                        self.left_framing,
                        self.sequence,
                        self.right_framing,
                    )
                else:
                    if left_framing:
                        other_extended_indices = (0,) + other_indices
                    else:
                        other_extended_indices = other_indices + (2,)
                    right_state_tuple = state_by_index(
                        other_extended_indices,
                        self.left_framing,
                        self.sequence,
                        self.right_framing,
                    )
                right_state = self.KLRW_algebra().state(right_state_tuple)
                braid = self.KLRW_algebra().braid(
                    state=right_state,
                    word=word,
                )

                sign = 1
                if ind == 1:
                    degree = sum(middle_part) % 2
                    if degree == 1:
                        sign *= -1

                chain_term = chain_map_dict[hom_degree]
                element = self.KLRW_algebra().term(braid, sign * coeff)
                if self_on_right:
                    chain_term[index_in_chain, other_index_in_chain] = element
                else:
                    chain_term[other_index_in_chain, index_in_chain] = element

        needs_corrections = self.needs_corrections or other.needs_corrections

        degree_shift = self.KLRW_algebra().grading_group.crossing_grading(
            framing_color,
            moving_color,
        )

        codomain = codomain[0, degree_shift]

        if needs_corrections:
            chain_map = chain_map_by_correction(
                chain_map_dict, domain, codomain, verbose=False, check=True,
            )
        else:
            chain_map = domain.hom(codomain, chain_map_dict)
        return chain_map.homology_class()

    """
    def _boundary_braided_crossing_morphism_from_exts_(
        self,
        left_framing: bool = True,
        self_on_right: bool = True,
    ):
        from sage.rings.rational_field import QQ

        # We need to work over a field,
        # include all the gradings,
        # and make sure inverse powers of
        # parameters do not show up.
        # Then the answer is unique.
        new_klrw_options = {
            "base_R": QQ,
            "vertex_scaling": True,
            "edge_scaling": True,
            "dot_scaling": True,
            "invertible_parameters": False,
        }
        old_klrw_options = self.KLRW_algebra()._reduction[2].copy()
        del old_klrw_options["quiver_data"]
        self_base_change = self._replace_(
            **new_klrw_options,
        )
        if left_framing:
            moving_color = self.sequence[0]
            framing_color = self.left_framing
            other_projective_state_tuple = (moving_color,)
            other_projective_state_tuple += (self.left_framing,)
            other_projective_state_tuple += self.sequence[1:]
            other_projective_state_tuple += (self.right_framing,)
        else:
            moving_color = self.sequence[-1]
            framing_color = self.right_framing
            other_projective_state_tuple = (self.left_framing,)
            other_projective_state_tuple += self.sequence[:-1]
            other_projective_state_tuple += (self.right_framing,)
            other_projective_state_tuple += (moving_color,)

        other_projective_state = self.KLRW_algebra().state(other_projective_state_tuple)
        projective = KLRWIrreducibleProjectiveModule(
            other_projective_state, 0, self.KLRW_algebra().grading_group
        )
        projective_base_change = KLRWIrreducibleProjectiveModule(
            other_projective_state, 0, self_base_change.KLRW_algebra().grading_group
        )
        other_kwrds = {
            "quiver": self.KLRW_algebra().quiver,
            "projective": projective,
            "elementary_transposition": self.braiding_sign,
        }
        other = BraidedProjective(**(old_klrw_options | other_kwrds))
        other_kwrds["projective"] = projective_base_change
        other_base_change = other._replace_(
            **(old_klrw_options | new_klrw_options | other_kwrds),
        )

        degree = self_base_change.KLRW_algebra().grading_group.crossing_grading(
            framing_color, moving_color
        )
        print(degree)
        range_ = ((0, 0),)
        # degree before base change
        original_degree = self.KLRW_algebra().grading_group(degree)
        print(original_degree)
        if self_on_right:
            exts, ext_dims = find_exts(
                other_base_change[0, -original_degree], self_base_change, range_=range_
            )
        else:
            exts, ext_dims = find_exts(
                self_base_change, other_base_change[0, original_degree], range_=range_
            )

        assert ext_dims[0, degree] <= 1, "Answer is not unique"
        assert ext_dims[0, degree] >= 1, "Answer does not exist"

        crossing_ext = exts[0, degree].basis()[0]

        return crossing_ext
    """

    @cached_method
    def _braided_dot_morphism_(
        self,
        dot_color: NodeInFramedQuiver,
        dot_position: int,
        self_on_right: bool = True,
    ):
        assert dot_position > 0
        stands_of_right_color = tuple(
            index for index, vertex in enumerate(self.sequence) if vertex == dot_color
        )
        assert dot_position <= len(stands_of_right_color)
        strand_with_dot_index = stands_of_right_color[dot_position - 1]

        index_by_multiindex = _index_by_multiindex_(len(self.sequence))

        chain_map_dict = {
            hom_deg: matrix(
                self.KLRW_algebra().opposite,
                self.component_rank(hom_deg),
                self.component_rank(hom_deg),
                sparse=True,
            )
            for hom_deg in self.gradings()
        }

        all_indices = tuple(range(3))
        iterator_pieces = [all_indices] * len(self.sequence)
        multi_index_iterator = product(*iterator_pieces)

        # make the geometric contribution
        for (*multiindex,) in multi_index_iterator:
            multiindex = tuple(multiindex)
            left_part = multiindex[:strand_with_dot_index]
            right_part = multiindex[strand_with_dot_index + 1 :]

            # degree = sum(multiindex)
            # sign = 1 if degree % 2 == 0 else -1
            # coeff = self.KLRW_algebra().base()(sign)

            homological_degree, index_in_chain = index_by_multiindex[multiindex]
            homological_degree *= -self.braiding_sign

            right_state_tuple = state_by_index(
                multiindex, self.left_framing, self.sequence, self.right_framing
            )
            right_state = self.KLRW_algebra().state(right_state_tuple)
            braid = self.KLRW_algebra().braid(
                state=right_state,
                word=(),
            )

            dot_position = self._position_among_same_color_(
                color=dot_color,
                index=multiindex[strand_with_dot_index],
                left_part=left_part,
                right_part=right_part,
            )

            x_dot = (
                self.KLRW_algebra()
                .base()
                .dot_variable(
                    dot_color,
                    dot_position,
                )
                .monomial
            )
            # coeff *= x_dot

            chain_term = chain_map_dict[homological_degree]
            element = self.KLRW_algebra().term(braid, x_dot)
            chain_term[index_in_chain, index_in_chain] = element

        # special_indices = ([0, 1], [0, 2], [1, 2])
        special_indices = ([1, 0], [2, 0], [2, 1])
        # != for boolean is xor
        if (self.braiding_sign == 1) != self_on_right:
            special_indices = tuple([b, a] for a, b in special_indices)

        multi_index_iterator = product(*iterator_pieces)
        relevant_positions = (
            pos
            for pos in range(len(self.sequence))
            if self.sequence[pos] == dot_color and pos != strand_with_dot_index
        )

        r_i = self.KLRW_algebra().base().vertex_variable(dot_color).monomial
        coeff = r_i

        # TODO: detailed description
        iterator_pieces = [relevant_positions]
        iterator_pieces += [all_indices] * (len(self.sequence) - 2)
        iterator_pieces += [special_indices]
        multi_index_iterator = product(*iterator_pieces)
        for pos, *common_indices, inds in multi_index_iterator:
            # print(pos, common_indices, inds)
            first_index, second_index = inds
            # print(pos, strand_with_dot_index)
            if pos < strand_with_dot_index:
                left_part = tuple(common_indices[:pos])
                middle_part = tuple(common_indices[pos : strand_with_dot_index - 1])
                right_part = tuple(common_indices[strand_with_dot_index - 1 :])
                sign = -1
            else:
                left_part = tuple(common_indices[:strand_with_dot_index])
                middle_part = tuple(common_indices[strand_with_dot_index : pos - 1])
                right_part = tuple(common_indices[pos - 1 :])
                sign = 1

            if 1 not in inds:
                # applying rule banning some combinations
                if 1 in middle_part:
                    continue
            else:
                # signs appear if 1's appear betweens swapping 1 and other index.
                degree = sum(middle_part) % 2
                if degree == 1:
                    sign *= -1

            # print(left_part, middle_part, right_part)

            self_indices = (
                left_part + (first_index,) + middle_part + (second_index,) + right_part
            )
            other_indices = (
                left_part + (second_index,) + middle_part + (first_index,) + right_part
            )

            # print(self_indices, other_indices)
            if first_index == 0:
                # only preceeding 0's shift the position
                self_first_position = sum(1 for x in left_part if x == 0)
                other_first_position = sum(1 for x in left_part if x == 0) + sum(
                    1 for x in middle_part if x == 0
                )
            elif first_index == 1:
                self_first_position = (
                    sum(1 for x in left_part if x == 0)
                    + sum(1 for x in middle_part if x != 2)
                    + sum(1 for x in right_part if x != 2)
                    + 1
                )
                if second_index != 2:
                    self_first_position += 1
                other_first_position = (
                    sum(1 for x in left_part if x == 0)
                    + sum(1 for x in middle_part if x == 0)
                    + sum(1 for x in right_part if x != 2)
                    + 1
                )
                if second_index != 2:
                    other_first_position += 1
            else:
                self_first_position = (
                    len(self.sequence)
                    + 1
                    - sum(1 for x in middle_part if x == 2)
                    - sum(1 for x in right_part if x == 2)
                )
                other_first_position = (
                    len(self.sequence) + 1 - sum(1 for x in right_part if x == 2)
                )

            if second_index == 0:
                # only preceeding 0's shift the position
                other_second_position = sum(1 for x in left_part if x == 0)
                self_second_position = sum(1 for x in left_part if x == 0) + sum(
                    1 for x in middle_part if x == 0
                )
            elif second_index == 1:
                other_second_position = (
                    sum(1 for x in left_part if x == 0)
                    + sum(1 for x in middle_part if x != 2)
                    + sum(1 for x in right_part if x != 2)
                    + 1
                )
                if first_index != 2:
                    other_second_position += 1
                self_second_position = (
                    sum(1 for x in left_part if x == 0)
                    + sum(1 for x in middle_part if x == 0)
                    + sum(1 for x in right_part if x != 2)
                    + 1
                )
                if first_index != 2:
                    self_second_position += 1
            else:
                other_second_position = (
                    len(self.sequence)
                    + 1
                    - sum(1 for x in middle_part if x == 2)
                    - sum(1 for x in right_part if x == 2)
                )
                self_second_position = (
                    len(self.sequence) + 1 - sum(1 for x in right_part if x == 2)
                )

            if self_on_right:
                first_word = KLRWbraid_set.word_for_one_strand(
                    right_moving_strand_position=self_first_position,
                    left_moving_strand_position=other_first_position,
                )
                second_word = KLRWbraid_set.word_for_one_strand(
                    right_moving_strand_position=self_second_position,
                    left_moving_strand_position=other_second_position,
                )
            else:
                first_word = KLRWbraid_set.word_for_one_strand(
                    right_moving_strand_position=other_first_position,
                    left_moving_strand_position=self_first_position,
                )
                second_word = KLRWbraid_set.word_for_one_strand(
                    right_moving_strand_position=other_second_position,
                    left_moving_strand_position=self_second_position,
                )
            # The strands do not intersect.
            # So word is just the concatination of two.
            # Which word comes first in lex min order depends
            # on which strand has lower index
            if self_first_position < self_second_position:
                word = first_word + second_word
            else:
                word = second_word + first_word

            hom_degree, index_in_chain = index_by_multiindex[self_indices]
            hom_degree *= -self.braiding_sign
            _, other_index_in_chain = index_by_multiindex[other_indices]

            if self_on_right:
                right_state_tuple = state_by_index(
                    self_indices, self.left_framing, self.sequence, self.right_framing
                )
            else:
                right_state_tuple = state_by_index(
                    other_indices, self.left_framing, self.sequence, self.right_framing
                )
            right_state = self.KLRW_algebra().state(right_state_tuple)
            braid = self.KLRW_algebra().braid(
                state=right_state,
                word=word,
            )

            chain_term = chain_map_dict[hom_degree]
            element = self.KLRW_algebra().term(braid, sign * coeff)

            if self_on_right:
                # chain_term[other_index_in_chain, index_in_chain] = element
                chain_term[index_in_chain, other_index_in_chain] = element
            else:
                # chain_term[index_in_chain, other_index_in_chain] = element
                chain_term[other_index_in_chain, index_in_chain] = element

        degree_shift = self.KLRW_algebra().grading_group.dot_algebra_grading(
            DotVariableIndex(
                vertex=dot_color,
                number=1,  # does not matter
            )
        )

        domain = self
        codomain = self

        codomain = codomain[0, degree_shift]

        if self.needs_corrections:
            chain_map = chain_map_by_correction(
                chain_map_dict, domain, codomain, verbose=False, check=True,
            )
        else:
            chain_map = domain.hom(codomain, chain_map_dict)
        return chain_map.homology_class()

    """
    def braided_crossing_morphism(
        self,
        i: int,
        self_on_right=True,
        only_maps=False,
    ):
        assert i >= 0
        assert i <= len(self.sequence)

        if i == 0:
            return self._boundary_braided_crossing_morphism_(
                ...
            )
        elif i == len(self.sequence):
            return
        else:
            return
    """


class BraidedProjective(KLRWPerfectComplex):
    homological_grading_group = StandardBraidedProjectives.homological_grading_group
    differential_degree = StandardBraidedProjectives.differential_degree

    @staticmethod
    def __classcall__(
        cls,
        quiver: FramedDynkinDiagram_class | CartanType_abstract,
        elementary_transposition: int,
        state: KLRWstate,
        base_R=ZZ,
        **klrw_options,
    ):
        return UniqueRepresentation.__classcall__(
            cls,
            base_R=base_R,
            quiver=quiver,
            state=state,
            elementary_transposition=elementary_transposition,
            **klrw_options,
        )

    def __init__(
        self,
        base_R,
        quiver: FramedDynkinDiagram_class | CartanType_abstract,
        state: KLRWstate,
        elementary_transposition: int,
        **klrw_options,
    ):
        """
        Construct a braided projective.

        We braid `i`-th and `i+1`-st framings, where
        `i = |elementary_transposition|`.
        If `i > 0` we braid in the positive direction,
        if `i < 0` we braid in the negative direction.

        Key feature:
            - if `BP_+` is a projective braided in the positive
            direction, then for any other projective `Q` we have
            `R^{k}Hom(BP_+, Q) = 0`
            unless `k = 0`.
            - if `BP_-` is a projective braided in the positive
            direction, then for any other projective `Q` we have
            `R^{k}Hom(Q, BP_-) = 0`
            unless `k = 0`.

        Note that the convention on the sign of braiding is the opposite
        for Ben Webster's. In his convention, `BP_+` is given by
        (derived) tensoring a projective with a bimodule, so it's
        quasi-isomorphic to an ordinary module. Then for
        projective `Q`s
        `R^{k}Hom(Q, BP_+) = 0`
        unless `k = 0`.
        Our convention is the opposite.
        TODO: synchronize conventions.
        """
        self.elementary_transposition = elementary_transposition
        assert self.elementary_transposition != 0
        if self.elementary_transposition > 0:
            sign = +1
        else:
            sign = -1
            self.elementary_transposition = -self.elementary_transposition

        self.base_R = base_R
        self.quiver = quiver
        self.state = state
        self.klrw_options = klrw_options

        framing_indices = tuple(
            index for index, vertex in enumerate(state) if vertex.is_framing()
        )
        assert self.elementary_transposition < len(framing_indices)
        left_framing_index = framing_indices[self.elementary_transposition - 1]
        right_framing_index = framing_indices[self.elementary_transposition]

        self.left_sequence = state[:left_framing_index].as_tuple()
        left_framing = state[left_framing_index]
        sequence = state[left_framing_index + 1 : right_framing_index].as_tuple()
        right_framing = state[right_framing_index]
        self.right_sequence = state[right_framing_index + 1 :].as_tuple()
        self.standard_complex = StandardBraidedProjectives(
            quiver,
            left_framing=left_framing,
            right_framing=right_framing,
            sequence=sequence,
            braiding_sign=sign,
        )

        self.left_strands_count = defaultdict(int)
        for v in self.left_sequence:
            self.left_strands_count[v] += 1
        self.left_and_center_strands_count = copy(self.left_strands_count)
        for v in sequence:
            self.left_and_center_strands_count[v] += 1

        standard_dots_algebra = self.standard_complex.KLRW_algebra().base()
        dots_algebra = self.KLRW_algebra().base()
        variables_images = [None] * standard_dots_algebra.ngens()
        for index, var in standard_dots_algebra.variables.items():
            if var.position is not None:
                if isinstance(index, DotVariableIndex):
                    vertex = index.vertex
                    new_index = DotVariableIndex(
                        vertex, index.number + self.left_strands_count[vertex]
                    )
                else:
                    new_index = index
                variables_images[var.position] = dots_algebra.variables[
                    new_index
                ].monomial

        self.hom_from_standard_dots = standard_dots_algebra.hom(
            variables_images, codomain=dots_algebra
        )

        # projectives = {
        #    hom_deg: [self.projective_from_standard(pr) for pr in projs]
        #    for hom_deg, projs in self.standard_complex.projectives_iter()
        # }

        # differential = {}
        # KLRW_algebra = self.KLRW_algebra()
        # for hom_deg, diff in self.standard_complex.differential.items():
        #    data = {
        #        (i, j): KLRW_algebra.sum(
        #            KLRW_algebra.term(
        #                KLRW_algebra.braid(
        #                    state=projectives[hom_deg - 1][i].state,
        #                    word=tuple(x + len(self.left_sequence) for x in braid.word()),
        #                ),
        #                self.hom_from_standard_dots(coeff),
        #            )
        #            for braid, coeff in entry.value
        #        )
        #        for (i, j), entry in diff.dict(copy=False).items()
        #    }
        #    differential[hom_deg] = matrix(
        #        KLRW_algebra.opposite,
        #        len(projectives[hom_deg - 1]),
        #        len(projectives[hom_deg]),
        #        data,
        #        sparse=True,
        #        immutable=True,
        #    )

        # super().__init__(
        #    ring=KLRW_algebra,
        #    projectives=projectives,
        #    differential=differential,
        #    differential_degree=self.standard_complex.differential.degree(),
        #    homological_grading_names=(None,),
        # )

    def framed_dynkin_with_dimensions(self):
        DD = FramedDynkinDiagram_with_dimensions.with_zero_dimensions(
            quiver=self.quiver
        )

        for v in self.state:
            DD[v] += 1

        return DD

    @lazy_attribute
    def _KLRW_algebra(self):
        DD = self.framed_dynkin_with_dimensions()

        return KLRWAlgebra(
            self.base_R,
            DD,
            **self.klrw_options,
        )

    @lazy_attribute
    def _extended_grading_group(self):
        # only one homological grading,
        # the default one
        return self._normalize_extended_grading_group_(
            ring=self._KLRW_algebra,
        )

    @lazy_attribute
    def needs_corrections(self):
        return self.standard_complex.needs_corrections

    @lazy_attribute
    def differential(self):
        differential = {}
        KLRW_algebra = self.KLRW_algebra()
        for hom_deg, diff in self.standard_complex.differential.items():
            data = {
                (i, j): KLRW_algebra.sum(
                    KLRW_algebra.term(
                        KLRW_algebra.braid(
                            state=self.projectives(hom_deg - 1)[i].state,
                            word=tuple(
                                x + len(self.left_sequence) for x in braid.word()
                            ),
                        ),
                        self.hom_from_standard_dots(coeff),
                    )
                    for braid, coeff in entry.value
                )
                for (i, j), entry in diff.dict(copy=False).items()
            }
            differential[hom_deg] = matrix(
                KLRW_algebra.opposite,
                self.component_rank(hom_deg - 1),
                self.component_rank(hom_deg),
                data,
                sparse=True,
                immutable=True,
            )

        return self.DifferentialClass(
            underlying_module=self,
            differential_data=differential,
            degree=self.differential_degree,
            check=True,
        )

    @lazy_attribute
    def _projectives(self):
        return {
            hom_deg: [self.projective_from_standard(pr) for pr in projs]
            for hom_deg, projs in self.standard_complex.projectives_iter()
        }

    def component_rank(self, grading):
        return self.standard_complex.component_rank(grading)

    def component_rank_iter(self):
        return self.standard_complex.component_rank_iter()

    def state_from_standard(self, state):
        return self.KLRW_algebra().state(
            self.left_sequence + state.as_tuple() + self.right_sequence
        )

    def projective_from_standard(self, projective: KLRWIrreducibleProjectiveModule):
        return KLRWIrreducibleProjectiveModule(
            self.state_from_standard(projective.state),
            projective.equivariant_degree,
            grading_group=self.KLRW_algebra().grading_group,
        )

    def braid_from_standard(self, braid):
        return self.KLRW_algebra().braid(
            state=self.state_from_standard(braid.state()),
            word=tuple(x + len(self.left_sequence) for x in braid.word()),
        )

    def element_from_standard(self, element):
        return self.KLRW_algebra().sum(
            self.KLRW_algebra().term(
                self.braid_from_standard(braid),
                self.hom_from_standard_dots(coeff),
            )
            for braid, coeff in element
        )

    def dot_to_standard(self, dot_index: DotVariableIndex) -> DotVariableIndex | None:
        """
        Remakes a dot label of the current dot algebra to the dot label
        in the standard complex.
        If the dot is not in the standard complex [i.e. corresponds to
        strands not between the braiding framings], then return `None`.
        """
        vertex = dot_index.vertex
        if (
            dot_index.number > self.left_and_center_strands_count[vertex]
            or dot_index.number <= self.left_strands_count[vertex]
        ):
            return None
        return DotVariableIndex(
            vertex, dot_index.number - self.left_strands_count[vertex]
        )

    def elementary_crossing_to_chain_map(self, i):
        """
        Make a chain map for an elementary crossing.

        The elementary crossing is the KLRW braid with
        right state `self.state` and length one word `(i,)`
        This element is considered as a map between projectives
        `left_state -> right_state`
        """
        assert i >= 0

        if i <= len(self.left_sequence) + 1:
            if i < len(self.left_sequence):
                return self._external_crossing_(i)
            else:
                inside_on_right = i != len(self.left_sequence)
                return self._boundary_crossing_(
                    left_framing=True,
                    inside_on_right=inside_on_right,
                )
        elif i <= len(self.left_sequence) + len(self.standard_complex.sequence) + 2:
            if i <= len(self.left_sequence) + len(self.standard_complex.sequence):
                return self._internal_crossing_(i)
            else:
                inside_on_right = (
                    i
                    == len(self.left_sequence) + len(self.standard_complex.sequence) + 1
                )
                return self._boundary_crossing_(
                    left_framing=False,
                    inside_on_right=inside_on_right,
                )
        else:
            return self._external_crossing_(i)

    def _domain_of_crossing_(self, i):
        """
        Making domain of the crossing morphism.
        """
        other_state = self.state.act_by_s(i)
        return self._replace_(state=other_state)

    def _equivariant_shift_of_crossing_(self, i):
        state = self.state
        equivariant_degree = self.equivariant_grading_group().crossing_grading(
            state[i - 1], state[i]
        )
        return equivariant_degree

    def _internal_crossing_(self, i):
        """
        A special case of elementary crossing.

        The crossing is between the braided framings.
        `self` with a shift is the codomain.
        """
        standard_chain = self.standard_complex._internal_braided_crossing_morphism_(
            i - len(self.left_sequence) - 1,
            self_on_right=True,
        )
        chain_map_dict = {}
        for hom_deg, map in standard_chain:
            chain_map_dict[hom_deg] = {
                (i, j): self.element_from_standard(entry.value)
                for (i, j), entry in map.dict(copy=False).items()
            }
        domain = self._domain_of_crossing_(i)
        codomain = self[0, self._equivariant_shift_of_crossing_(i)]
        chain_map = domain.hom(codomain, chain_map_dict, check=False)
        return chain_map.homology_class()

    def _boundary_crossing_(self, left_framing: bool, inside_on_right: bool):
        """
        A special case of elementary crossing.

        The crossing involves one of the braided framings.
        `self` is codomain.
        If `inside_on_right = True`, then the moving strand is
        between the bradied framings on the right.

        `self` with a shift is the codomain.
        """
        # index of the crossing
        i = len(self.left_sequence) + 1
        if not left_framing:
            i += len(self.standard_complex.sequence)
        if not inside_on_right:
            if left_framing:
                i -= 1
            else:
                i += 1

        domain = self._domain_of_crossing_(i)

        chain_map_dict = {}
        if inside_on_right:
            standard_chain = self.standard_complex._boundary_braided_crossing_morphism_(
                left_framing,
                self_on_right=True,
            )
            for hom_deg, map in standard_chain:
                chain_map_dict[hom_deg] = {
                    (i, j): self.element_from_standard(entry.value)
                    for (i, j), entry in map.dict(copy=False).items()
                }
        else:
            standard_chain = (
                domain.standard_complex._boundary_braided_crossing_morphism_(
                    left_framing,
                    self_on_right=False,
                )
            )
            for hom_deg, map in standard_chain:
                chain_map_dict[hom_deg] = {
                    (i, j): domain.element_from_standard(entry.value)
                    for (i, j), entry in map.dict(copy=False).items()
                }

        codomain = self[0, self._equivariant_shift_of_crossing_(i)]
        chain_map = domain.hom(codomain, chain_map_dict, check=False)
        return chain_map.homology_class()

    def _external_crossing_(self, i):
        """
        A special case of elementary crossing.

        The crossing is outside of the braided framings.

        `self` with a shift is the codomain.
        """
        chain_map_dict = {}
        for hom_deg, projs in self.standard_complex.projectives_iter():
            chain_map_dict[hom_deg] = {
                (j, j): self.KLRW_algebra().KLRWmonomial(
                    state=self.state_from_standard(projs[j].state), word=(i,)
                )
                for j in range(len(projs))
            }

        domain = self._domain_of_crossing_(i)
        codomain = self[0, self._equivariant_shift_of_crossing_(i)]
        chain_map = domain.hom(codomain, chain_map_dict, check=False)

        return chain_map.homology_class()

    def word_to_chain_map(self, word):
        if not word:
            return self.hom_set(self).one()

        result = self.elementary_crossing_to_chain_map(word[-1])
        intermediate_domain = self

        # iterating in reverse order over all
        # elements except the last one,
        # since it's already taken into account.
        for i in range(len(word) - 2, -1, -1):
            intermediate_domain = intermediate_domain._domain_of_crossing_(word[i + 1])
            result = result * intermediate_domain.elementary_crossing_to_chain_map(
                word[i]
            )

        return result

    def dot_to_chain_map(self, dot_index: DotVariableIndex):
        equivariant_shift = self.equivariant_grading_group().dot_algebra_grading(
            dot_index
        )
        codomain = self[0, equivariant_shift]

        standard_dot_index = self.dot_to_standard(dot_index)
        chain_map_dict = {}
        if standard_dot_index is None:
            coeff = self.KLRW_algebra().base().variables[dot_index].monomial
            for hom_deg, projs in self.standard_complex.projectives_iter():
                chain_map_dict[hom_deg] = {
                    (j, j): self.KLRW_algebra().term(
                        self.KLRW_algebra().braid(
                            state=self.state_from_standard(projs[j].state),
                            word=(),
                        ),
                        coeff=coeff,
                    )
                    for j in range(len(projs))
                }
        else:
            standard_chain = self.standard_complex._braided_dot_morphism_(
                dot_color=standard_dot_index.vertex,
                dot_position=standard_dot_index.number,
                self_on_right=True,
            )
            for hom_deg, map in standard_chain:
                chain_map_dict[hom_deg] = {
                    (i, j): self.element_from_standard(entry.value)
                    for (i, j), entry in map.dict(copy=False).items()
                }

        chain_map = self.hom(codomain, chain_map_dict, check=False)
        return chain_map.homology_class()

    @classmethod
    def klrw_element_to_chain_map(cls, element, elementary_transposition):
        assert not element.is_zero()
        klrw_options = element.parent()._reduction[2].copy()
        base_R = klrw_options.pop("base_R")
        quiver = klrw_options.pop("quiver_data").quiver
        right_state = element.right_state(check_if_all_have_same_right_state=True)
        BP_right = cls(
            base_R=base_R,
            quiver=quiver,
            state=right_state,
            elementary_transposition=elementary_transposition,
            **klrw_options,
        )
        left_state = element.left_state(check_if_all_have_same_left_state=False)
        BP_left = cls(
            base_R=base_R,
            quiver=quiver,
            state=left_state,
            elementary_transposition=elementary_transposition,
            **klrw_options,
        )
        dot_algebra = BP_right.KLRW_algebra().base()
        result = None
        for braid, coeff in element:
            braid_part = BP_right.word_to_chain_map(braid.word())

            coeff_part = BP_left.hom_set(BP_left).zero()
            for cent_poly, dict_of_dots in dot_algebra._dict_of_dots_iterator_(coeff):
                if dict_of_dots:
                    dot_part = prod(
                        (
                            BP_left.dot_to_chain_map(dot_index) ** pow
                            for dot_index, pow in dict_of_dots
                        )
                    )
                else:
                    dot_part = BP_left.hom_set(BP_left).one()

                piece = cent_poly * dot_part
                coeff_part += piece

            # if we read braid as a map from left to right,
            # when dots act first because they are on the left
            # and the map that acts first should be on the right
            if result is None:
                result = braid_part * coeff_part
            else:
                result += braid_part * coeff_part

        return result

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
        """
        Change the scalars.

        Warning: it does not change other options of KLRW algebra.
        Use `meth:_replace_` instead.
        """
        return self._replace_(
            base_R=other,
        )
