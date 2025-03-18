# cython: profile=True

import cython
import numpy as np

from scipy.sparse import dok_matrix, csc_matrix

from sage.matrix.matrix0 import Matrix

from klrw.klrw_algebra import KLRWAlgebra
from klrw.gradings import HomologicalGradingGroupElement
from klrw.dot_algebra import KLRWUpstairsDotsAlgebra
from klrw.perfect_complex import KLRWPerfectComplex
from cython.cimports.klrw.cython_exts.sparse_csc import CSC_Mat
from cython.cimports.klrw.cython_exts.sparse_csr import CSR_Mat
from cython.cimports.klrw.cython_exts.perfect_complex_corrections_ext import (
    CorrectionsMatrix,
)


@cython.ccall
def system_d_h_piece(
    klrw_algebra: KLRWAlgebra,
    d_shifted_codomain_csc: CSC_Mat,
    h_csc: CorrectionsMatrix,
    basis_appearing_in_product: dict,
    relevant_parameter_part,
):
    """
    Computes
    `-codomain_differential * morphism`
    where `morphism` runs through a basis in the space
    of possible homotopies.
    """
    end_algebra = klrw_algebra.opposite
    dot_algebra = klrw_algebra.base()
    system_piece = dok_matrix(
        (len(basis_appearing_in_product), h_csc.number_of_corrections()),
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

    # need to iterate over the list of all corrections stored in h_csc

    M: cython.int = len(h_csc.indptrs) - 1
    for j in range(M):
        for indptr1 in range(h_csc.indptrs[j], h_csc.indptrs[j + 1]):
            # intermediate index in matrix product
            k = h_csc.indices[indptr1]
            # now we iterate over all enries in k-th column in d_geom
            # and all variables in (k,j)-th entry of h
            for variable_number in range(
                h_csc.entry_ptrs[indptr1], h_csc.entry_ptrs[indptr1 + 1]
            ):
                correction_braid = h_csc.corrections_braid[variable_number]
                correction_exp = h_csc.corrections_exp[variable_number]
                correction = klrw_algebra.term(
                    correction_braid, dot_algebra.monomial(*correction_exp)
                )
                correction = end_algebra(correction)
                for indptr0 in range(
                    d_shifted_codomain_csc.indptrs[k],
                    d_shifted_codomain_csc.indptrs[k + 1],
                ):
                    i = d_shifted_codomain_csc.indices[indptr0]
                    d_entry = d_shifted_codomain_csc.data[indptr0]
                    product = -d_entry * correction
                    for braid, poly in product.value:
                        for exp, scalar in poly.iterator_exp_coeff():
                            if relevant_parameter_part is not None:
                                if dot_algebra.etuple_ignoring_dots(exp) != relevant_parameter_part:
                                    continue
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
def system_h_d_piece(
    klrw_algebra: KLRWAlgebra,
    d_domain_csr: CSR_Mat,
    h_csc: CorrectionsMatrix,
    basis_appearing_in_product: dict,
    relevant_parameter_part,
):
    """
    Computes
    `morphism * domain_differential`
    where `morphism` runs through a basis in the space
    of possible homotopies.
    """
    dot_algebra = klrw_algebra.base()
    end_algebra = klrw_algebra.opposite
    system_piece = dok_matrix(
        (len(basis_appearing_in_product), h_csc.number_of_corrections()),
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

    # need to iterate over the list of all corrections stored in h_csc

    M: cython.int = len(h_csc.indptrs) - 1
    for k in range(M):
        for indptr1 in range(h_csc.indptrs[k], h_csc.indptrs[k + 1]):
            # intermediate index in matrix product
            i = h_csc.indices[indptr1]
            # now we iterate over all enries in k-th column in d_geom
            # and all variables in (k,j)-th entry of h
            for variable_number in range(
                h_csc.entry_ptrs[indptr1], h_csc.entry_ptrs[indptr1 + 1]
            ):
                correction_braid = h_csc.corrections_braid[variable_number]
                correction_exp = h_csc.corrections_exp[variable_number]
                correction = klrw_algebra.term(
                    correction_braid, dot_algebra.monomial(*correction_exp)
                )
                correction = end_algebra(correction)
                for indptr0 in range(
                    d_domain_csr.indptrs[k], d_domain_csr.indptrs[k + 1]
                ):
                    j = d_domain_csr.indices[indptr0]
                    d_entry = d_domain_csr.data[indptr0]
                    product = correction * d_entry
                    for braid, poly in product.value:
                        for exp, scalar in poly.iterator_exp_coeff():
                            if relevant_parameter_part is not None:
                                if dot_algebra.etuple_ignoring_dots(exp) != relevant_parameter_part:
                                    continue
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
def system_rhs_piece(
    image_chain_map: Matrix,
    basis_appearing_in_product: dict,
    dot_algebra: KLRWUpstairsDotsAlgebra,
    relevant_parameter_part,
):
    system_piece = dok_matrix(
        (len(basis_appearing_in_product), 1),
        dtype="intc",
    )

    i: cython.int
    j: cython.int

    for ij, entry in image_chain_map.dict(copy=False).items():
        i, j = ij
        for braid, poly in entry.value:
            for exp, scalar in poly.iterator_exp_coeff():
                if relevant_parameter_part is not None:
                    if dot_algebra.etuple_ignoring_dots(exp) != relevant_parameter_part:
                        continue
                key = (i, j, braid.word(), exp)
                result_index: cython.int
                assert key in basis_appearing_in_product, "Inconsistent system:" + repr(
                    key
                )
                result_index = basis_appearing_in_product[key]

                system_piece[result_index, 0] = scalar

    return system_piece


@cython.ccall
def homotopy_piece(
    klrw_algebra: KLRWAlgebra,
    h_piece_csc: CorrectionsMatrix,
    x_piece_csc: csc_matrix,
):
    assert x_piece_csc.shape[1] == 1, "x must be a column"

    end_algebra = klrw_algebra.opposite
    dot_algebra = klrw_algebra.base()

    homotopy_piece: dict = {}

    j: cython.int = 0
    i: cython.int
    ptr_x: cython.int = 0
    var_number: cython.int = x_piece_csc.indices[0]

    first_var_in_next_column: cython.int
    for j in range(len(h_piece_csc.indptrs) - 1):
        first_var_in_next_column = h_piece_csc.entry_ptrs[h_piece_csc.indptrs[j + 1]]
        while var_number < first_var_in_next_column:
            # find entry with the currect variable
            entry_ptrs_in_this_column = h_piece_csc.entry_ptrs[
                h_piece_csc.indptrs[j] : h_piece_csc.indptrs[j + 1] + 1
            ]
            next_entry_index = np.searchsorted(
                entry_ptrs_in_this_column,
                var_number,
                side="right",
            )
            first_var_number_in_next_entry = entry_ptrs_in_this_column[next_entry_index]
            # the zero index in the slice corresponds
            # to h_piece_csc.indptrs[j] in the original array
            entry_index: cython.int = next_entry_index + h_piece_csc.indptrs[j] - 1
            # sum over all variables in the same matrix element
            entry_element = klrw_algebra.zero()
            while var_number < first_var_number_in_next_entry:
                braid = h_piece_csc.corrections_braid[var_number]
                exp = h_piece_csc.corrections_exp[var_number]
                scalar = x_piece_csc.data[ptr_x]
                poly = scalar * dot_algebra.monomial(*exp)
                entry_element += klrw_algebra.term(braid, poly)

                ptr_x += 1
                if ptr_x < len(x_piece_csc.indices):
                    var_number = x_piece_csc.indices[ptr_x]
                else:
                    var_number = x_piece_csc.shape[0]

            i = h_piece_csc.indices[entry_index]
            homotopy_piece[i, j] = end_algebra(entry_element)

    return homotopy_piece


@cython.ccall
def make_homotopy_dict(
    klrw_algebra: KLRWAlgebra,
    h_csc: dict[HomologicalGradingGroupElement, CorrectionsMatrix],
    x_csc: csc_matrix,
    ordered_term_degrees: list[HomologicalGradingGroupElement],
):
    homotopy_dict = {}
    index_begin_in_hom_deg: cython.int = 0
    for hom_deg in ordered_term_degrees:
        index_end_in_hom_deg = index_begin_in_hom_deg + len(
            h_csc[hom_deg]._corrections_braid()
        )
        # making a copy of the piece corresponding to this homological degree
        x_csc_piece = x_csc[index_begin_in_hom_deg:index_end_in_hom_deg, :]
        index_begin_in_hom_deg = index_end_in_hom_deg
        if x_csc_piece.nnz == 0:
            continue

        homotopy_dict[hom_deg] = homotopy_piece(
            klrw_algebra=klrw_algebra,
            h_piece_csc=h_csc[hom_deg],
            x_piece_csc=x_csc_piece,
        )

    return homotopy_dict
