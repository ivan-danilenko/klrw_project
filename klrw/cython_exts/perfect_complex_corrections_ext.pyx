# cython: profile=True

import cython
import numpy as np

from types import MappingProxyType

# from cython.cimports.libcpp.vector import queue

from scipy.sparse import dok_matrix, csr_matrix, csc_matrix, spmatrix

from sage.rings.polynomial.polydict import ETuple
from sage.rings.integer_ring import ZZ
from sage.structure.element import Element


from klrw.klrw_algebra import KLRWAlgebra
from klrw.gradings import (
    QuiverGradingGroupElement,
    QuiverGradingGroup,
)
from klrw.dot_algebra import KLRWUpstairsDotsAlgebra
from cython.cimports.klrw.cython_exts.sparse_csc import CSC_Mat
from cython.cimports.klrw.cython_exts.sparse_csr import CSR_Mat
from klrw.cython_exts.sparse_addition import add


@cython.cclass
class CorrectionsMatrix:
    """
    To record all corrections we have a data structure similar
    to CSR matrices.
    indptrs and indices are the same as in CSR matrices:
    the i-th row is [self.indptrs[i]:self.indptrs[i+1]]
    in self.indices and self.entry_ptrs.
    Let a be in range(self.indptrs[i], self.indptrs[i+1])
    Then self.indices[a] keeps the number of column of an entry;
    Possible corrections are in listed in
    self.corrections_...[self.entry_ptrs[a]:self.entry_ptrs[a+1]]
    corrections_braid gives the braid
    corrections_exp gives the ETuple of exponents in the dot algebra
    """

    corrections_braid: object[::1]
    corrections_exp: object[::1]
    entry_ptrs: cython.int[::1]
    indices: cython.int[::1]
    indptrs: cython.int[::1]
    number_of_rows: cython.int

    def __init__(
        self,
        corrections_braid,
        corrections_exp,
        entry_ptrs,
        indices,
        indptrs,
        number_of_rows,
    ):
        assert len(entry_ptrs) == len(indices) + 1, (
            repr(len(entry_ptrs)) + " " + repr(len(indices))
        )
        assert indptrs[0] == 0
        assert indptrs[-1] == len(indices), repr(indptrs[-1]) + " " + repr(len(indices))
        assert entry_ptrs[-1] == len(corrections_braid), (
            repr(entry_ptrs[-1]) + " " + repr(len(corrections_braid))
        )
        assert len(corrections_braid) == len(corrections_exp), (
            repr(len(corrections_braid)) + " " + repr(len(corrections_exp))
        )

        self.corrections_braid = corrections_braid
        self.corrections_exp = corrections_exp
        self.entry_ptrs = entry_ptrs
        self.indices = indices
        self.indptrs = indptrs
        self.number_of_rows = number_of_rows

    @cython.ccall
    def _corrections_braid(self):
        return self.corrections_braid

    @cython.ccall
    def _corrections_exp(self):
        return self.corrections_exp

    @cython.ccall
    def _entry_ptrs(self):
        return self.entry_ptrs

    @cython.ccall
    def _indices(self):
        return self.indices

    @cython.ccall
    def _indptrs(self):
        return self.indptrs

    @cython.ccall
    def _number_of_columns(self):
        return len(self.indptrs) - 1

    @cython.ccall
    def _number_of_rows(self):
        return self.number_of_rows

    def __reduce__(self):
        return self.__class__, (
            np.asarray(self.corrections_braid),
            np.asarray(self.corrections_exp),
            np.asarray(self.entry_ptrs),
            np.asarray(self.indices),
            np.asarray(self.indptrs),
            self.number_of_rows,
        )

    def number_of_corrections(self):
        return len(self.corrections_braid)

    def dict(self):
        i: cython.int
        j: cython.int
        k: cython.int

        d = {}

        for i in range(len(self.indptrs) - 1):
            for j in range(self.indptrs[i], self.indptrs[i + 1]):
                d[self.indices[j], i] = []
                for k in range(self.entry_ptrs[j], self.entry_ptrs[j + 1]):
                    entry = (self.corrections_braid[k], self.corrections_exp[k])
                    d[self.indices[j], i].append(entry)

        return d


@cython.ccall
def system_d_geom_d1_piece(
    klrw_algebra: KLRWAlgebra,
    d_geom_csc: CSC_Mat,
    d1_csc: CorrectionsMatrix,
    relevant_coeff_degree: QuiverGradingGroupElement,
    basis_appearing_in_product: dict,
):
    end_algebra = klrw_algebra.opposite
    dot_algebra = klrw_algebra.base()
    grading_group = klrw_algebra.grading_group
    system_piece = dok_matrix(
        (len(basis_appearing_in_product), d1_csc.number_of_corrections()),
        dtype="intc",
    )

    i: cython.int
    j: cython.int
    k: cython.int
    indptr0: cython.int
    indptr1: cython.int
    variable_number: cython.int

    # basis in product is given by a quadruple
    # (i, j, braid, exp), where
    # i is a row index
    # j is a column index
    # braid is a KLRW braid
    # exp is an ETuple of exponents in the dot algebra

    # need to iterate over the list of all corrections stored in d1_csc

    M: cython.int = len(d1_csc.indptrs) - 1
    for j in range(M):
        for indptr1 in range(d1_csc.indptrs[j], d1_csc.indptrs[j + 1]):
            # intermediate index in matrix product
            k = d1_csc.indices[indptr1]
            # now we iterate over all enries in k-th column in d_geom
            # and all variables in (k,j)-th entry of d1
            for variable_number in range(
                d1_csc.entry_ptrs[indptr1], d1_csc.entry_ptrs[indptr1 + 1]
            ):
                correction_braid = d1_csc.corrections_braid[variable_number]
                correction_exp = d1_csc.corrections_exp[variable_number]
                correction = klrw_algebra.term(
                    correction_braid, dot_algebra.monomial(*correction_exp)
                )
                correction = end_algebra(correction)
                for indptr0 in range(d_geom_csc.indptrs[k], d_geom_csc.indptrs[k + 1]):
                    i = d_geom_csc.indices[indptr0]
                    d_geom_entry = d_geom_csc.data[indptr0]
                    product = d_geom_entry * correction
                    for braid, poly in product.value:
                        for exp, scalar in poly.iterator_exp_coeff():
                            if dot_algebra.exp_degree(exp, grading_group) == relevant_coeff_degree:
                                key = (i, j, braid.word(), exp)
                                result_index: cython.int
                                if key in basis_appearing_in_product:
                                    result_index = basis_appearing_in_product[key]
                                else:
                                    result_index = len(basis_appearing_in_product)
                                    basis_appearing_in_product[key] = result_index
                                    # need to add one more row
                                    system_piece.resize(
                                        result_index + 1, system_piece.shape[1]
                                    )

                                system_piece[result_index, variable_number] = scalar

    return system_piece


@cython.ccall
def system_d1_d_geom_piece(
    klrw_algebra: KLRWAlgebra,
    d_geom_csr: CSR_Mat,
    d1_csc: CorrectionsMatrix,
    relevant_coeff_degree: QuiverGradingGroupElement,
    basis_appearing_in_product: dict,
):
    dot_algebra = klrw_algebra.base()
    end_algebra = klrw_algebra.opposite
    grading_group = klrw_algebra.grading_group
    system_piece = dok_matrix(
        (len(basis_appearing_in_product), d1_csc.number_of_corrections()),
        dtype="intc",
    )

    i: cython.int
    j: cython.int
    k: cython.int
    indptr0: cython.int
    indptr1: cython.int
    variable_number: cython.int


    # basis in product is given by a quadruple
    # (i, j, braid, exp), where
    # i is a row index
    # j is a column index
    # braid is a KLRW braid
    # exp is an ETuple of exponents in the dot algebra

    # need to iterate over the list of all corrections stored in d1_csc

    M: cython.int = len(d1_csc.indptrs) - 1
    for k in range(M):
        for indptr1 in range(d1_csc.indptrs[k], d1_csc.indptrs[k + 1]):
            # intermediate index in matrix product
            i = d1_csc.indices[indptr1]
            # now we iterate over all enries in k-th column in d_geom
            # and all variables in (k,j)-th entry of d1
            for variable_number in range(
                d1_csc.entry_ptrs[indptr1], d1_csc.entry_ptrs[indptr1 + 1]
            ):
                correction_braid = d1_csc.corrections_braid[variable_number]
                correction_exp = d1_csc.corrections_exp[variable_number]
                correction = klrw_algebra.term(
                    correction_braid, dot_algebra.monomial(*correction_exp)
                )
                correction = end_algebra(correction)
                for indptr0 in range(d_geom_csr.indptrs[k], d_geom_csr.indptrs[k + 1]):
                    j = d_geom_csr.indices[indptr0]
                    d_geom_entry = d_geom_csr.data[indptr0]
                    product = correction * d_geom_entry
                    for braid, poly in product.value:
                        for exp, scalar in poly.iterator_exp_coeff():
                            if dot_algebra.exp_degree(exp, grading_group) == relevant_coeff_degree:
                                key = (i, j, braid.word(), exp)
                                result_index: cython.int
                                if key in basis_appearing_in_product:
                                    result_index = basis_appearing_in_product[key]
                                else:
                                    result_index = len(basis_appearing_in_product)
                                    basis_appearing_in_product[key] = result_index
                                    # need to add one more row
                                    system_piece.resize(
                                        result_index + 1, system_piece.shape[1]
                                    )

                                system_piece[result_index, variable_number] = scalar

    return system_piece


@cython.ccall
def system_d_squared_piece(
    dot_algebra: KLRWUpstairsDotsAlgebra,
    grading_group: QuiverGradingGroup,
    d_csr: CSR_Mat,
    d_csc: CSC_Mat,
    relevant_coeff_degree: QuiverGradingGroupElement,
    basis_appearing_in_product: dict,
):
    assert d_csr._number_of_columns() == d_csc._number_of_rows(), (
        repr(d_csr._number_of_columns()) + " != " + repr(d_csc._number_of_rows())
    )

    system_piece = dok_matrix(
        (len(basis_appearing_in_product), 1),
        dtype="intc",
    )

    M_number_of_rows: cython.int = len(d_csr.indptrs) - 1
    N_number_of_columns: cython.int = len(d_csc.indptrs) - 1

    i: cython.int
    j: cython.int
    indptr1: cython.int
    indptr2: cython.int
    indptr1_end: cython.int
    indptr2_end: cython.int

    entry_can_be_non_zero: cython.bint = False

    for i in range(M_number_of_rows):
        for j in range(N_number_of_columns):
            indptr1 = d_csr.indptrs[i]
            indptr2 = d_csc.indptrs[j]
            indptr1_end = d_csr.indptrs[i + 1]
            indptr2_end = d_csc.indptrs[j + 1]
            while indptr1 != indptr1_end and indptr2 != indptr2_end:
                if d_csr.indices[indptr1] == d_csc.indices[indptr2]:
                    if not entry_can_be_non_zero:
                        dot_product = d_csr.data[indptr1] * d_csc.data[indptr2]
                        entry_can_be_non_zero = True
                    else:
                        dot_product += d_csr.data[indptr1] * d_csc.data[indptr2]
                    indptr1 += 1
                    indptr2 += 1
                elif d_csr.indices[indptr1] < d_csc.indices[indptr2]:
                    indptr1 += 1
                else:
                    indptr2 += 1

            if entry_can_be_non_zero:
                if dot_product:
                    for braid, poly in dot_product.value:
                        for exp, scalar in poly.iterator_exp_coeff():
                            if dot_algebra.exp_degree(exp, grading_group) == relevant_coeff_degree:
                                key = (i, j, braid.word(), exp)
                                result_index: cython.int
                                assert (
                                    key in basis_appearing_in_product
                                ), "Inconsistent system:" + repr(key)
                                result_index = basis_appearing_in_product[key]

                                system_piece[result_index, 0] = scalar
                entry_can_be_non_zero = False

    return system_piece


@cython.ccall
def correction_piece_csc(
    klrw_algebra,
    d1_piece_csc: CorrectionsMatrix,
    x_piece_csc: csc_matrix,
    projectives_left,
    projectives_right,
):
    assert x_piece_csc.shape[1] == 1, "x must be a column"

    end_algebra = klrw_algebra.opposite
    dot_algebra = klrw_algebra.base()

    corretion_csc_indptrs: cython.int[::1] = np.zeros(
        len(d1_piece_csc.indptrs), dtype=np.dtype("intc")
    )
    max_non_zero_entries: cython.int = x_piece_csc.nnz
    corretion_csc_indices: cython.int[::1] = np.zeros(
        max_non_zero_entries, dtype=np.dtype("intc")
    )
    corretion_csc_data: object[::1] = np.empty(max_non_zero_entries, dtype=object)

    j: cython.int = 0
    ptr_x: cython.int = 0
    var_number: cython.int = x_piece_csc.indices[0]

    non_zero_entries_so_far: cython.int = 0

    first_var_in_next_column: cython.int
    for j in range(len(d1_piece_csc.indptrs) - 1):
        first_var_in_next_column = d1_piece_csc.entry_ptrs[d1_piece_csc.indptrs[j + 1]]
        while var_number < first_var_in_next_column:
            # find entry with the currect variable
            entry_ptrs_in_this_column = d1_piece_csc.entry_ptrs[
                d1_piece_csc.indptrs[j] : d1_piece_csc.indptrs[j + 1] + 1
            ]
            next_entry_index = np.searchsorted(
                entry_ptrs_in_this_column,
                var_number,
                side="right",
            )
            first_var_number_in_next_entry = entry_ptrs_in_this_column[next_entry_index]
            # the zero index in the slice corresponds
            # to d1_piece_csc.indptrs[j] in the original array
            entry_index: cython.int = next_entry_index + d1_piece_csc.indptrs[j] - 1
            # sum over all variables in the same matrix element
            entry_element = klrw_algebra.zero()
            while var_number < first_var_number_in_next_entry:
                braid = d1_piece_csc.corrections_braid[var_number]
                exp = d1_piece_csc.corrections_exp[var_number]
                scalar = x_piece_csc.data[ptr_x]
                poly = scalar * dot_algebra.monomial(*exp)
                entry_element += klrw_algebra.term(braid, poly)

                ptr_x += 1
                if ptr_x < len(x_piece_csc.indices):
                    var_number = x_piece_csc.indices[ptr_x]
                else:
                    var_number = x_piece_csc.shape[0]

            corretion_csc_indices[non_zero_entries_so_far] = d1_piece_csc.indices[
                entry_index
            ]
            elem = entry_element
            assert (
                elem.right_state(check_if_all_have_same_right_state=True)
                == projectives_right[d1_piece_csc.indices[entry_index]].state
            )
            assert (
                elem.left_state(check_if_all_have_same_left_state=True)
                == projectives_left[j].state
            )

            corretion_csc_data[non_zero_entries_so_far] = end_algebra(elem)

            non_zero_entries_so_far += 1

        corretion_csc_indptrs[j + 1] = non_zero_entries_so_far

    corretion_csc_indices = corretion_csc_indices[:non_zero_entries_so_far]
    corretion_csc_data = corretion_csc_data[:non_zero_entries_so_far]
    return CSC_Mat(
        corretion_csc_data,
        corretion_csc_indices,
        corretion_csc_indptrs,
        d1_piece_csc.number_of_rows,
    )


@cython.ccall
def update_differential(
    klrw_algebra,
    d_geom_csc: dict[Element, CSC_Mat] | MappingProxyType[Element, CSC_Mat],
    d1_csc: (
        dict[Element, CorrectionsMatrix] | MappingProxyType[Element, CorrectionsMatrix]
    ),
    x_csc: csc_matrix,
    projectives,
    degree,
):
    corrected_differential = {}
    index_begin_in_hom_deg = 0
    for hom_deg in sorted(d_geom_csc.keys()):
        index_end_in_hom_deg = index_begin_in_hom_deg + len(
            d1_csc[hom_deg]._corrections_braid()
        )
        # making a copy of the piece corresponding to this homological degree
        x_csc_piece = x_csc[index_begin_in_hom_deg:index_end_in_hom_deg, :]
        if x_csc_piece.nnz == 0:
            corrected_differential[hom_deg] = d_geom_csc[hom_deg]
        else:
            corr_piece_csc = correction_piece_csc(
                klrw_algebra,
                d1_csc[hom_deg],
                x_csc_piece,
                projectives[hom_deg],
                projectives[hom_deg + degree],
            )
            corrected_differential[hom_deg] = add(d_geom_csc[hom_deg], corr_piece_csc)

        index_begin_in_hom_deg = index_end_in_hom_deg

    return corrected_differential


def min_exp_in_product(left: CSR_Mat, right: CSC_Mat) -> ETuple | None:
    assert (
        left.number_of_columns == right.number_of_rows
    ), "Shapes do not match: the number of columns in left does not match the number of columns in right."

    min_exp: ETuple | None = None

    left_number_of_rows: cython.int = len(left.indptrs) - 1
    right_number_of_columns: cython.int = len(right.indptrs) - 1

    i: cython.int
    j: cython.int
    indptr1: cython.int
    indptr2: cython.int
    indptr1_end: cython.int
    indptr2_end: cython.int

    entry_can_be_non_zero: cython.bint = False

    for i in range(left_number_of_rows):
        for j in range(right_number_of_columns):
            indptr1 = left.indptrs[i]
            indptr2 = right.indptrs[j]
            indptr1_end = left.indptrs[i + 1]
            indptr2_end = right.indptrs[j + 1]
            while indptr1 != indptr1_end and indptr2 != indptr2_end:
                if left.indices[indptr1] == right.indices[indptr2]:
                    if not entry_can_be_non_zero:
                        dot_product = left.data[indptr1] * right.data[indptr2]
                        entry_can_be_non_zero = True
                    else:
                        dot_product += left.data[indptr1] * right.data[indptr2]
                    indptr1 += 1
                    indptr2 += 1
                elif left.indices[indptr1] < right.indices[indptr2]:
                    indptr1 += 1
                else:
                    indptr2 += 1

            if entry_can_be_non_zero:
                for _, poly in dot_product.value:
                    for exp in poly.exponents():
                        if min_exp is None:
                            min_exp = exp
                        elif exp < min_exp:
                            min_exp = exp
                entry_can_be_non_zero = False

    return min_exp
