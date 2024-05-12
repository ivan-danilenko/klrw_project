from collections import defaultdict
from itertools import product
from functools import cache

from sage.matrix.constructor import matrix
from sage.functions.other import Function_factorial

from .framed_dynkin import NodeInFramedQuiverWithMarks
from .perfect_complex import KLRWProjectiveModule
from .perfect_complex_corrections import PerfectComplex


def terms_in_homology_degree(n, k):
    """
    n choose k
    """
    factorial = Function_factorial()
    return factorial(n) // (factorial(n - k) * factorial(k))


def maps_to(index):
    """
    Generator returning (sign,next_index)
    Sign is the Koszul sign,
    next_index is the index where one of the zeroes is replaced by a one
    """
    iter = (x for x in range(len(index)) if index[x] == 1)
    for i in iter:
        left_part = index[:i]
        right_part = index[i + 1 :]
        degree = sum(right_part)
        sign = 1 if degree % 2 == 0 else -1
        yield sign, left_part + (0,) + right_part


def braid_from_marked_state(left_state, right_state):
    """
    We assume that both states have marks that make each strand type unique
    """
    # make reduced word by finding elements of the Lehmer cocode
    rw = []
    permutation = {
        left_index: right_state.index(mark)
        for left_index, mark in enumerate(left_state)
    }
    for left_index, mark in enumerate(left_state):
        right_index = permutation[left_index]
        cocode_element = sum(
            1 for i in range(left_index) if permutation[i] > right_index
        )
        piece = [
            j + 1 for j in range(left_index - 1, left_index - cocode_element - 1, -1)
        ]
        rw += piece

    word = tuple(rw)
    return word


@cache
def state_by_index(ind, left_framing, left_sequence, right_sequence, right_framing):
    state = [
        left_sequence[x] for x in range(len(left_sequence) - 1, -1, -1) if ind[x] == 0
    ]
    state += [left_framing]
    state += [left_sequence[x] for x in range(len(left_sequence)) if ind[x] == 1]
    state += [
        right_sequence[x]
        for x in range(len(right_sequence))
        if ind[x + len(left_sequence)] == 0
    ]
    state += [right_framing]
    state += [
        right_sequence[x]
        for x in range(len(right_sequence) - 1, -1, -1)
        if ind[x + len(left_sequence)] == 1
    ]
    return tuple(state)


@cache
def unmark(iterable):
    return iterable.__class__(i.unmark() for i in iterable)


def stable_envelope(
    KLRW,
    left_framing,
    right_framing,
    sequence,
    dots_on_left,
):
    """
    TODO: Make independent of global variables
    """

    Braid = KLRW.KLRWBraid
    State = Braid.KLRWstate_set
    State.enable_checks()
    differentials = {}
    projectives = defaultdict(list)

    index_in_chain_by_multi_index = {}
    equivariant_degree_by_multi_index = {}

    # mark moving strands to distinguish them
    sequence = tuple(
        NodeInFramedQuiverWithMarks(node, i) for i, node in enumerate(sequence)
    )

    left_framing = NodeInFramedQuiverWithMarks(left_framing, 0)
    right_framing = NodeInFramedQuiverWithMarks(right_framing, 1)

    left_sequence = sequence[:dots_on_left]
    right_sequence = sequence[dots_on_left:]

    multi_index_iterator = product(*([range(2)] * len(sequence)))
    # The zeroeth term we treat differently
    multiindex = next(multi_index_iterator)
    marked_state = state_by_index(
        multiindex, left_framing, left_sequence, right_sequence, right_framing
    )
    state = State._element_constructor_(unmark(marked_state))
    index_in_chain_by_multi_index[multiindex] = 0
    equivariant_degree_by_multi_index[multiindex] = KLRW.grading_group.zero()
    projectives[0] = [KLRWProjectiveModule(state, KLRW.grading_group.zero())]

    for multiindex in multi_index_iterator:
        homological_degree = sum(multiindex)
        index_in_chain = len(projectives[homological_degree])
        index_in_chain_by_multi_index[multiindex] = index_in_chain
        if index_in_chain == 0:
            differentials[homological_degree] = matrix(
                KLRW,
                terms_in_homology_degree(len(sequence), homological_degree),
                terms_in_homology_degree(len(sequence), homological_degree - 1),
                sparse=True,
            )

        diff = differentials[homological_degree]

        marked_state = state_by_index(
            multiindex, left_framing, left_sequence, right_sequence, right_framing
        )
        state = State._element_constructor_(unmark(marked_state))

        for sign, new_index in maps_to(multiindex):
            new_marked_state = state_by_index(
                new_index, left_framing, left_sequence, right_sequence, right_framing
            )
            word = braid_from_marked_state(marked_state, new_marked_state)
            new_state = State._element_constructor_(unmark(new_marked_state))
            braid = Braid._element_constructor_(new_state, word)
            assert braid.left_state() == state
            entry = KLRW.base()(sign) * KLRW.monomial(braid)
            if multiindex not in equivariant_degree_by_multi_index:
                equivariant_degree_by_multi_index[multiindex] = (
                    equivariant_degree_by_multi_index[new_index]
                    - entry.degree(check_if_homogeneous=True)
                )
            else:
                assert equivariant_degree_by_multi_index[
                    multiindex
                ] == equivariant_degree_by_multi_index[new_index] - entry.degree(
                    check_if_homogeneous=True
                )
            codomain_index_in_chain = index_in_chain_by_multi_index[new_index]
            diff[index_in_chain, codomain_index_in_chain] = entry

        projectives[homological_degree].append(
            KLRWProjectiveModule(state, equivariant_degree_by_multi_index[multiindex])
        )

    try:
        stab = PerfectComplex(
            KLRW,
            differentials=differentials,
            projectives=projectives,
            degree=-1,
        )
    except AssertionError:
        return differentials, projectives

    return stab
