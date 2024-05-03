# cython: profile=True

import cython
import numpy as np

from typing import Iterable

from sage.rings.integer import Integer

from klrw.klrw_algebra import KLRWAlgebra
from cython.cimports.klrw.cython_exts.sparse_csc import CSC_Mat


@cython.cfunc
def strands_to_the_left(all_points: Iterable, point: cython.int) -> cython.int:
    res: cython.int = 0
    for pt in all_points:
        if pt < point:
            res += 1
    return res


@cython.ccall
def d_times_one_piece(
    klrw_algebra: KLRWAlgebra,
    d_curr: CSC_Mat,
    curr_right_thimbles: list,
    curr_left_thimbles: list,
    next_thimbles: list,
    right_projectives: list,
    left_projectives: list,
    hom_add_one_more_strand: dict,
    sign=Integer(1),
) -> CSC_Mat:
    """
    Generate a block matrix.

    Each block corresponds to an element of d_curr
    The block is "identity*element"
    Here we write "" because it's not a product,
    it's merging of two pictures of moving strands.
    Identity is in the vector space associated with
    next_right_thimbles
    """
    assert len(right_projectives) == len(curr_right_thimbles) * len(next_thimbles)
    assert len(left_projectives) == len(curr_left_thimbles) * len(next_thimbles)
    assert len(curr_left_thimbles) == d_curr.number_of_rows
    assert len(curr_right_thimbles) == len(d_curr.indptrs) - 1

    output_indptrs = np.zeros(
        len(curr_right_thimbles) * len(next_thimbles) + 1,
        dtype=np.dtype("intc"),
    )
    max_number_of_entries = d_curr.nnz() * len(next_thimbles)
    output_indices = np.zeros(
        max_number_of_entries,
        dtype=np.dtype("intc"),
    )
    output_data = np.empty(
        max_number_of_entries,
        dtype=np.dtype("O"),
    )

    V, _ = klrw_algebra.quiver.vertices()
    number_of_moving_strands = klrw_algebra.quiver[V]
    dot_algebra = klrw_algebra.base()
    grading_group = klrw_algebra.grading_group

    j: cython.int
    indptr: cython.int
    dimension_in_next: cython.int = len(next_thimbles)

    entries_so_far: cython.int = 0
    for j in range(len(d_curr.indptrs) - 1):
        right_thimble_curr = curr_right_thimbles[j]
        for k in range(dimension_in_next):
            column_index = j * dimension_in_next + k
            segment_next = next_thimbles[k].colored_state[0]
            point_next = next_thimbles[k].points[0]
            # find the number of moving strands to the left
            right_number_of_next_strand = strands_to_the_left(
                right_thimble_curr.points, point_next
            )
            # add the number of framing strands to the left
            right_position_of_new_strand = right_number_of_next_strand + segment_next
            right_proj = right_projectives[column_index]
            for indptr in range(d_curr.indptrs[j], d_curr.indptrs[j + 1]):
                entry_curr = d_curr.data[indptr]
                row_curr = d_curr.indices[indptr]
                row_index = row_curr * dimension_in_next + k
                left_proj = left_projectives[row_index]

                left_thimble_curr = curr_left_thimbles[row_curr]
                left_number_of_next_strand = strands_to_the_left(
                    left_thimble_curr.points, point_next
                )
                left_position_of_new_strand = left_number_of_next_strand + segment_next

                # see conventions in CombinatorialEBrane.__init__()
                hom_to_more_dots = hom_add_one_more_strand[
                    number_of_moving_strands - 1, left_number_of_next_strand + 1
                ]

                entry = klrw_algebra.zero()
                for braid, coef in entry_curr:
                    # prepare the new braid by adding one more strand
                    mapping = {}
                    for r, vert in enumerate(right_proj.state):
                        if vert == V:
                            if r != right_position_of_new_strand:
                                if r > right_position_of_new_strand:
                                    r_cur = r - 1
                                else:
                                    r_cur = r
                                l = braid.find_position_on_other_side(
                                    r_cur, reverse=True
                                )
                                if l >= left_position_of_new_strand:
                                    l += 1
                                mapping[l] = r

                    d_times_one_braid = (
                        klrw_algebra.braid_set().braid_by_extending_permutation(
                            right_state=right_proj.state,
                            mapping=mapping,
                        )
                    )
                    new_coeff = hom_to_more_dots(coef)
                    if not new_coeff.is_zero():
                        braid_degree = klrw_algebra.braid_degree(d_times_one_braid)
                        coeff_degree = dot_algebra.element_degree(
                            new_coeff, grading_group=grading_group, check_if_homogeneous=True
                        )
                        term_degree = braid_degree + coeff_degree

                        if (
                            right_proj.equivariant_degree - left_proj.equivariant_degree
                            == term_degree
                        ):
                            entry += klrw_algebra.term(d_times_one_braid, new_coeff)

                entry *= sign
                if not entry.is_zero():
                    assert (
                        entry.left_state(check_if_all_have_same_left_state=True)
                        == left_proj.state
                    ), (
                        repr(entry.left_state(check_if_all_have_same_left_state=True))
                        + " != "
                        + repr(left_proj.state)
                    )

                    output_indices[entries_so_far] = row_index
                    output_data[entries_so_far] = entry
                    entries_so_far += 1

            output_indptrs[column_index + 1] = entries_so_far

    # Deleting tails of None's in data and zeroes in indices
    output_data = output_data[:entries_so_far]
    output_indices = output_indices[:entries_so_far]

    return CSC_Mat(
        output_data,
        output_indices,
        output_indptrs,
        len(left_projectives),
    )


@cython.ccall
def one_times_d_piece(
    klrw_algebra: KLRWAlgebra,
    d_next: CSC_Mat,
    curr_thimbles: list,
    next_right_thimbles: list,
    next_left_thimbles: list,
    right_projectives: list,
    left_projectives: list,
    hom_one_to_many_dots: dict,
    sign=Integer(1),
) -> CSC_Mat:
    """
    Generate a block diagonal matrix.

    Each block is d_next with elements modified by
    adding more strands
    """
    assert len(right_projectives) == len(curr_thimbles) * len(next_right_thimbles)
    assert len(left_projectives) == len(curr_thimbles) * len(next_left_thimbles)
    assert len(next_left_thimbles) == d_next.number_of_rows
    assert len(next_right_thimbles) == len(d_next.indptrs) - 1

    output_indptrs = np.zeros(
        len(next_right_thimbles) * len(curr_thimbles) + 1,
        dtype=np.dtype("intc"),
    )
    max_number_of_entries = d_next.nnz() * len(curr_thimbles)
    output_indices = np.zeros(
        max_number_of_entries,
        dtype=np.dtype("intc"),
    )
    output_data = np.empty(
        max_number_of_entries,
        dtype=np.dtype("O"),
    )

    V, _ = klrw_algebra.quiver.vertices()
    number_of_moving_strands = klrw_algebra.quiver[V]
    dot_algebra = klrw_algebra.base()
    grading_group = klrw_algebra.grading_group

    j: cython.int
    indptr: cython.int
    column_dimension_in_next: cython.int = len(d_next.indptrs) - 1
    row_dimension_in_next: cython.int = d_next.number_of_rows

    entries_so_far: cython.int = 0
    for k in range(len(curr_thimbles)):
        thimble_curr = curr_thimbles[k]
        for j in range(len(d_next.indptrs) - 1):
            right_segment_next = next_right_thimbles[j].colored_state[0]
            right_point_next = next_right_thimbles[j].points[0]
            column_index = k * column_dimension_in_next + j
            # find the number of moving strands to the left
            right_number_of_next_strand = strands_to_the_left(
                thimble_curr.points, right_point_next
            )
            # add the number of framing strands to the left
            right_position_of_new_strand = (
                right_number_of_next_strand + right_segment_next
            )
            right_proj = right_projectives[column_index]
            for indptr in range(d_next.indptrs[j], d_next.indptrs[j + 1]):
                entry_next = d_next.data[indptr]
                row_next = d_next.indices[indptr]
                row_index = k * row_dimension_in_next + row_next
                left_proj = left_projectives[row_index]

                left_segment_next = next_left_thimbles[row_next].colored_state[0]
                left_point_next = next_left_thimbles[row_next].points[0]
                left_number_of_next_strand = strands_to_the_left(
                    thimble_curr.points, left_point_next
                )
                left_position_of_new_strand = (
                    left_number_of_next_strand + left_segment_next
                )

                # see conventions in CombinatorialEBrane.__init__()
                hom_to_more_dots = hom_one_to_many_dots[
                    number_of_moving_strands - 1, left_number_of_next_strand + 1
                ]

                entry = klrw_algebra.zero()
                assert len(entry_next) == 1
                for braid, coef in entry_next:
                    # prepare the new braid by adding one more strand
                    mapping = {
                        left_position_of_new_strand: right_position_of_new_strand
                    }

                    d_times_one_braid = (
                        klrw_algebra.braid_set().braid_by_extending_permutation(
                            right_state=right_proj.state,
                            mapping=mapping,
                        )
                    )
                    new_coeff = hom_to_more_dots(coef)
                    if not new_coeff.is_zero():
                        braid_degree = klrw_algebra.braid_degree(d_times_one_braid)
                        coeff_degree = dot_algebra.element_degree(
                            new_coeff, grading_group=grading_group, check_if_homogeneous=True
                        )
                        term_degree = braid_degree + coeff_degree

                        if (
                            right_proj.equivariant_degree - left_proj.equivariant_degree
                            == term_degree
                        ):
                            entry += klrw_algebra.term(d_times_one_braid, new_coeff)

                entry *= sign
                if not entry.is_zero():
                    assert (
                        entry.left_state(check_if_all_have_same_left_state=True)
                        == left_proj.state
                    ), (
                        repr(entry.left_state(check_if_all_have_same_left_state=True))
                        + " != "
                        + repr(left_proj.state)
                    )

                    output_indices[entries_so_far] = row_index
                    output_data[entries_so_far] = entry
                    entries_so_far += 1

            output_indptrs[column_index + 1] = entries_so_far

    # Deleting tails of None's in data and zeroes in indices
    output_data = output_data[:entries_so_far]
    output_indices = output_indices[:entries_so_far]

    return CSC_Mat(
        output_data,
        output_indices,
        output_indptrs,
        len(left_projectives),
    )


@cython.ccall
def CSC_from_dict_of_blocks(matrix_blocks: dict[CSC_Mat]):
    """
    Create a CSC block matrix.

    The (i,j)-th block is matrix_blocks[i,j]
    The indices i and j are ordered by standard comparison.
    They can be any comparable elements. In particular,
    they can be tuples of integers, negative integers, etc
    """
    blocks_dict_of_dicts = {}
    column_block_sizes = {}
    row_block_sizes = {}
    number_of_non_zeros = 0
    for i, j in matrix_blocks.keys():
        block: CSC_Mat = matrix_blocks[i, j]
        number_of_non_zeros += block.nnz()

        # if we have already seen this row
        if i in row_block_sizes:
            assert (
                row_block_sizes[i] == block.number_of_rows
            ), "Blocks have inconsistent size."
        else:
            row_block_sizes[i] = block.number_of_rows

        # if we have already seen this column
        if j in column_block_sizes:
            assert (
                column_block_sizes[j] == len(block.indptrs) - 1
            ), "Blocks have inconsistent size."
            blocks_dict_of_dicts[j][i] = block
        else:
            column_block_sizes[j] = len(block.indptrs) - 1
            blocks_dict_of_dicts[j] = {}
            blocks_dict_of_dicts[j][i] = block

    row_separations = {}
    row_number = 0
    for i in sorted(row_block_sizes.keys()):
        row_separations[i] = row_number
        row_number += row_block_sizes[i]

    number_of_columns = sum(column_block_sizes.values())
    output_indptrs = np.zeros(
        number_of_columns + 1,
        dtype=np.dtype("intc"),
    )
    output_indices = np.zeros(
        number_of_non_zeros,
        dtype=np.dtype("intc"),
    )
    output_data = np.empty(
        number_of_non_zeros,
        dtype=np.dtype("O"),
    )

    j_in_block: cython.int
    indptr: cython.int
    column_number: cython.int = 0
    entries_so_far: cython.int = 0
    for j in sorted(blocks_dict_of_dicts.keys()):
        column_blocks = blocks_dict_of_dicts[j]
        row_indices_in_column_sorted = sorted(column_blocks.keys())
        for j_in_block in range(column_block_sizes[j]):
            for i in row_indices_in_column_sorted:
                block: CSC_Mat = matrix_blocks[i, j]
                for indptr in range(
                    block.indptrs[j_in_block], block.indptrs[j_in_block + 1]
                ):
                    output_data[entries_so_far] = block.data[indptr]
                    output_indices[entries_so_far] = (
                        block.indices[indptr] + row_separations[i]
                    )
                    entries_so_far += 1
            output_indptrs[column_number + 1] = entries_so_far
            column_number += 1

    return CSC_Mat(
        output_data,
        output_indices,
        output_indptrs,
        row_number,
    )
