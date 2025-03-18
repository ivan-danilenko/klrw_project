from types import MappingProxyType
from typing import Iterable
import multiprocessing as mp
import numpy as np
from itertools import starmap

from scipy.sparse import block_array

from sage.structure.element import Element
from sage.matrix.matrix0 import Matrix
from sage.matrix.constructor import matrix
from sage.rings.polynomial.polydict import ETuple
from sage.rings.integer_ring import ZZ

from .klrw_algebra import KLRWAlgebra
from klrw.cython_exts.sparse_csc import CSC_Mat
from klrw.cython_exts.sparse_csr import CSR_Mat

from .perfect_complex import (
    KLRWPerfectComplex,
    KLRWIrreducibleProjectiveModule,
)
from klrw.cython_exts.perfect_complex_corrections_ext import (
    CorrectionsMatrix,
    system_d_geom_d1_piece,
    system_d1_d_geom_piece,
    system_d_squared_piece,
    update_differential,
    min_exp_in_product,
)


def PerfectComplex(
    ring: KLRWAlgebra,
    differential: dict[Element, object],
    projectives: dict[Element, Iterable[KLRWIrreducibleProjectiveModule]] | None = None,
    differential_degree=ZZ(-1),
    parallel_processes=8,
    max_iterations_for_corrections=10,
    verbose=True,
):
    if max_iterations_for_corrections == 0:
        return KLRWPerfectComplex(
            ring=ring,
            projectives=projectives,
            differential=differential,
            differential_degree=differential_degree,
        )

    d_csc = corrected_diffirential_csc(
        klrw_algebra=ring,
        differential=differential,
        projectives=projectives,
        degree=differential_degree,
        parallel_processes=parallel_processes,
        max_iterations_for_corrections=max_iterations_for_corrections,
        verbose=verbose,
    )

    diff = {
        hom_deg: matrix(
            ring.opposite,
            len(projectives[hom_deg + differential_degree]),
            len(projectives[hom_deg]),
            mat.dict(),
            sparse=True,
        )
        for hom_deg, mat in d_csc.items()
    }

    return KLRWPerfectComplex(
        ring=ring,
        projectives=projectives,
        differential=diff,
        differential_degree=differential_degree,
    )


def corrected_diffirential_csc(
    klrw_algebra: KLRWAlgebra,
    differential: dict[Element, object] | MappingProxyType[Element, object],
    projectives: dict[Element, Iterable[KLRWIrreducibleProjectiveModule]] | None = None,
    degree=-1,
    parallel_processes=1,
    max_iterations_for_corrections=10,
    verbose=True,
):
    d_csc = {}
    d_csr = {}
    end_algebra = klrw_algebra.opposite
    for hom_deg, mat in differential.items():
        if isinstance(mat, CSC_Mat):
            d_csc[hom_deg] = mat
        elif isinstance(mat, CSR_Mat):
            d_csc[hom_deg] = mat.to_csc()
        elif isinstance(mat, dict):
            d_csc[hom_deg] = CSC_Mat.from_dict(
                mat,
                number_of_rows=len(projectives[hom_deg + degree]),
                number_of_columns=len(projectives[hom_deg]),
            )
        elif isinstance(mat, Matrix):
            d_csc[hom_deg] = CSC_Mat.from_dict(
                mat.dict(),
                number_of_rows=len(projectives[hom_deg + degree]),
                number_of_columns=len(projectives[hom_deg]),
            )
        else:
            raise ValueError("Unknown type of differential matrices")

        d_csc[hom_deg].change_ring(end_algebra)

        if isinstance(mat, CSR_Mat):
            d_csr[hom_deg] = mat.change_ring(end_algebra)
        else:
            d_csr[hom_deg] = d_csc[hom_deg].to_csr()

    exp = min_exp_in_square(d_csr, d_csc, degree=degree)
    if exp is None:
        if verbose:
            print("Differential closes!")
        return d_csc

    def geometric_part(entry):
        return end_algebra(entry.value.geometric_part())

    d_geom_csc = {
        hom_deg: mat.apply_entrywise(geometric_part, inplace=False).eliminate_zeros()
        for hom_deg, mat in d_csc.items()
    }
    d_geom_csr = {hom_deg: mat.to_csr() for hom_deg, mat in d_geom_csc.items()}

    # MappingProxyType to prevent accident modification
    d_geom_csc = MappingProxyType(d_geom_csc)
    d_geom_csr = MappingProxyType(d_geom_csr)

    dot_algebra = klrw_algebra.base()
    ignore = None
    exp = dot_algebra.etuple_ignoring_dots(exp)
    assert (
        not exp.is_constant()
    ), "The initial differential does not square to zero mod extra variables."

    print(">>>Correcting the differential!<<<")

    for k in range(max_iterations_for_corrections):
        if verbose:
            param_monomial = dot_algebra.monomial(*exp)
            print("Correcting terms with {}".format(param_monomial))

        corrections = possible_corrections(
            klrw_algebra,
            projectives,
            degree=degree,
            relevant_parameter_part=exp,
            parallel_processes=parallel_processes,
            ignore=ignore,
            verbose=verbose,
        )

        basis_in_product = {}
        lhs = system_on_corrections_lhs(
            klrw_algebra=klrw_algebra,
            d_geom_csc=d_geom_csc,
            d_geom_csr=d_geom_csr,
            projectives=projectives,
            corrections=corrections,
            basis_appearing_in_product=basis_in_product,
            degree=degree,
            relevant_parameter_part=exp,
            verbose=verbose,
        )

        rhs = system_on_corrections_rhs(
            dot_algebra=klrw_algebra.base(),
            grading_group=klrw_algebra.grading_group,
            d_csc=d_csc,
            d_csr=d_csr,
            projectives=projectives,
            basis_appearing_in_product=basis_in_product,
            degree=degree,
            relevant_parameter_part=exp,
            verbose=verbose,
        )

        x = solve_system(lhs, rhs, verbose=verbose)

        d_csc = update_differential(
            klrw_algebra,
            d_csc,
            corrections,
            x,
            projectives,
            degree,
        )
        d_csr = {key: mat.to_csr() for key, mat in d_csc.items()}

        exp = min_exp_in_square(d_csr, d_csc, degree=degree)
        if exp is None:
            if verbose:
                print("Differential closes!")
            break
        exp = dot_algebra.etuple_ignoring_dots(exp)
    else:
        raise RuntimeError(
            "Could not make differential closed in under {} iterations".format(
                max_iterations_for_corrections
            )
        )

    return d_csc


def possible_corrections(
    klrw_algebra,
    projectives,
    degree,
    relevant_parameter_part,
    parallel_processes=1,
    ignore: dict[frozenset] | None = None,
    verbose=True,
):
    if ignore is None:
        ignore = {hom_deg: frozenset() for hom_deg in projectives}

    # terms of differential exist only between adjacent coh degrees
    possible_corrections_hom_degrees = frozenset(
        hom_deg for hom_deg in projectives if hom_deg + degree in projectives
    )
    tasks = [
        (
            klrw_algebra,
            projectives[hom_deg],
            projectives[hom_deg + degree],
            relevant_parameter_part,
            hom_deg,
            degree,
            ignore[hom_deg],
            verbose,
        )
        for hom_deg in possible_corrections_hom_degrees
    ]
    if parallel_processes == 1:
        corrections = dict(starmap(possible_corrections_in_hom_degree_and_print, tasks))
    else:
        mp_context = mp.get_context("spawn")
        # manager = mp_context.Manager()
        # corrections = manager.dict()
        with mp_context.Pool(processes=parallel_processes) as pool:
            corrections_async_list = pool.starmap_async(
                possible_corrections_in_hom_degree_and_print, tasks
            )
            corrections = dict(corrections_async_list.get())

    return corrections


def possible_corrections_in_hom_degree_and_print(
    klrw_algebra: KLRWAlgebra,
    projectives_domain: list,
    projectives_codomain: list,
    relevant_parameter_part,
    hom_deg,
    degree,
    ignore: frozenset = frozenset(),
    verbose=True,
):
    if verbose:
        print(
            "Start: Finding possible corrections for C_{} -> C_{}".format(
                hom_deg, hom_deg + degree
            )
        )

    corrections_in_hom_deg = possible_corrections_in_hom_degree(
        klrw_algebra=klrw_algebra,
        projectives_domain=projectives_domain,
        projectives_codomain=projectives_codomain,
        relevant_parameter_part=relevant_parameter_part,
        ignore=ignore,
    )

    if verbose:
        print(
            "End: Finding possible corrections for C_{} -> C_{}".format(
                hom_deg, hom_deg + degree
            )
        )

    return hom_deg, corrections_in_hom_deg


def possible_corrections_in_hom_degree(
    klrw_algebra: KLRWAlgebra,
    projectives_domain: list,
    projectives_codomain: list,
    relevant_parameter_part,
    ignore: frozenset = frozenset(),
):
    # To record all corrections we have a data structure similar
    # to CSR matrices.
    # indptrs and indices are the same as in CSR matrices:
    # the i-th row is [d1_csc_indptrs[i]:d1_csc_indptrs[i+1]]
    # in d1_csc_indices and d1_csc_entry_ptrs.
    # Let a be in range(d1_csc_indptrs[i], d1_csc_indptrs[i+1])
    # Then d1_csc_indices[a] keeps the number of column of an entry;
    # Possible corrections are in listed in
    # d1_csc_corrections[d1_csc_entry_ptrs[a]:d1_csc_entry_ptrs[a+1]]
    #
    # At first we make lists, then use numpy arrays.
    d1_csc_corrections_braid_list = []
    d1_csc_corrections_exp_list = []
    d1_csc_entry_ptrs_list = [0]
    d1_csc_indices_list = []
    d1_csc_indptrs = np.zeros(len(projectives_domain) + 1, dtype=np.dtype("intc"))
    entry_index = 0
    variable_index = 0
    # left_index is column index
    # right_index is row index
    for left_index, left_projective in enumerate(projectives_domain):
        for right_index, right_projective in enumerate(projectives_codomain):
            if (right_index, left_index) in ignore:
                continue

            equ_degree = (
                right_projective.equivariant_degree - left_projective.equivariant_degree
            )
            graded_component = klrw_algebra[
                left_projective.state : right_projective.state : equ_degree
            ]
            basis = graded_component.basis(
                relevant_parameter_part=relevant_parameter_part,
                as_tuples=True,
            )

            if not basis:
                continue

            for braid, exp in basis:
                d1_csc_corrections_braid_list.append(braid)
                d1_csc_corrections_exp_list.append(exp)
                variable_index += 1
            d1_csc_entry_ptrs_list.append(variable_index)
            d1_csc_indices_list.append(right_index)
            entry_index += 1

        d1_csc_indptrs[left_index + 1] = entry_index

    d1_csc_corrections_braid = np.array(
        d1_csc_corrections_braid_list, dtype=np.dtype("O")
    )
    # to avoid making a 2d array
    d1_csc_corrections_exp = np.empty(
        len(d1_csc_corrections_exp_list), dtype=np.dtype("O")
    )
    for i in range(len(d1_csc_corrections_exp)):
        d1_csc_corrections_exp[i] = d1_csc_corrections_exp_list[i]
    d1_csc_entry_ptrs = np.array(d1_csc_entry_ptrs_list, dtype=np.dtype("intc"))
    d1_csc_indices = np.array(d1_csc_indices_list, dtype=np.dtype("intc"))

    return CorrectionsMatrix(
        d1_csc_corrections_braid,
        d1_csc_corrections_exp,
        d1_csc_entry_ptrs,
        d1_csc_indices,
        d1_csc_indptrs,
        len(projectives_codomain),
    )


def system_on_corrections_lhs(
    klrw_algebra: KLRWAlgebra,
    d_geom_csc: dict[CSC_Mat],
    d_geom_csr: dict[CSR_Mat],
    projectives: dict[list],
    corrections: dict[CorrectionsMatrix],
    basis_appearing_in_product: dict,
    degree,
    relevant_parameter_part,
    verbose=True,
):
    matrix_blocks = [[None] * len(projectives) for _ in range(len(projectives))]
    # To have a good block-band matrix
    # hom_deg + degree has to be right before or
    # right after hom_deg in the sorted key list.
    # To allow degree = +/-1, we use this
    hom_deg_to_index = {
        hom_deg: index for index, hom_deg in enumerate(sorted(projectives.keys()))
    }
    for hom_deg in sorted(projectives.keys()):

        if hom_deg + degree not in hom_deg_to_index:
            continue

        if verbose:
            print(
                "Making the left hand side for the corrections C_{} -> C_{}".format(
                    hom_deg, hom_deg + 2 * degree
                )
            )
        hom_deg_index = hom_deg_to_index[hom_deg]
        hom_deg_next_index = hom_deg_to_index[hom_deg + degree]
        basis_appearing_in_product[hom_deg] = {}
        basis_dict = basis_appearing_in_product[hom_deg]
        if hom_deg + degree in corrections and hom_deg in d_geom_csc:
            matrix_blocks[hom_deg_index][hom_deg_index] = system_d_geom_d1_piece(
                klrw_algebra=klrw_algebra,
                d_geom_csc=d_geom_csc[hom_deg + degree],
                d1_csc=corrections[hom_deg],
                relevant_parameter_part=relevant_parameter_part,
                basis_appearing_in_product=basis_dict,
            )
        if hom_deg in corrections and hom_deg + degree in d_geom_csc:
            matrix_blocks[hom_deg_index][hom_deg_next_index] = system_d1_d_geom_piece(
                klrw_algebra=klrw_algebra,
                d_geom_csr=d_geom_csr[hom_deg],
                d1_csc=corrections[hom_deg + degree],
                relevant_parameter_part=relevant_parameter_part,
                basis_appearing_in_product=basis_dict,
            )
            if matrix_blocks[hom_deg_index][hom_deg_index] is not None:
                A = matrix_blocks[hom_deg_index][hom_deg_index]
                B = matrix_blocks[hom_deg_index][hom_deg_next_index]
                A.resize(B.shape[0], A.shape[1])

    A = block_array(matrix_blocks, format="csr")

    return A


def system_on_corrections_rhs(
    dot_algebra,
    grading_group,
    d_csc: dict[CSC_Mat],
    d_csr: dict[CSR_Mat],
    projectives: dict[list],
    basis_appearing_in_product: dict,
    degree,
    relevant_parameter_part,
    verbose=True,
):
    augment_blocks = [[None] for _ in range(len(projectives))]
    # To have a good block-band matrix
    # hom_deg + degree has to be right before or
    # right after hom_deg in the sorted key list.
    # To allow degree = +/-1, we use this
    hom_deg_to_index = {
        hom_deg: index for index, hom_deg in enumerate(sorted(projectives.keys()))
    }
    for hom_deg in sorted(projectives.keys()):

        if hom_deg + degree not in hom_deg_to_index:
            continue

        if verbose:
            print(
                "Making the right hand side for the corrections C_{} -> C_{}".format(
                    hom_deg, hom_deg + 2 * degree
                )
            )
        hom_deg_index = hom_deg_to_index[hom_deg]
        basis_dict = basis_appearing_in_product[hom_deg]
        if hom_deg + degree in d_csr and hom_deg in d_csc:
            augment_blocks[hom_deg_index] = [
                system_d_squared_piece(
                    dot_algebra,
                    grading_group,
                    d_csr=d_csr[hom_deg + degree],
                    d_csc=d_csc[hom_deg],
                    relevant_parameter_part=relevant_parameter_part,
                    basis_appearing_in_product=basis_dict,
                )
            ]

    b = block_array(augment_blocks, format="csc")
    b *= -1

    return b


def solve_system(A, b, verbose=True):
    import gurobipy as gp

    # a bug in Gurobi since 12.0.1:
    # subtraction is not supported with scipy's sparse matrices,
    # but addition is supported.
    # So, we use addition instead of == that does subtraction.
    y_neg = -b.transpose()

    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as m:
            x = m.addMVar(A.shape[1], lb=-float("inf"), vtype="I")
            m.addConstr(A @ x + y_neg == 0)
            # set method to primal simplex method
            m.Params.Method = 0
            m.optimize()
            # if model is infeasible
            if m.Status == 3:
                m.computeIIS()
                m.write("model.ilp")
            x = x.X

    from scipy.sparse import csr_matrix

    # making a sparse integer-valued vector from x
    # we make it sparse because most entries are expected to be zero
    x = csr_matrix(x, dtype="intc")
    x = x.transpose()

    assert (A @ x - b).nnz == 0, "Integer approximation is not a solution!"

    return x


def min_exp_in_square(
    d_csr: dict[Element, CSR_Mat],
    d_csc: dict[Element, CSC_Mat],
    degree,
) -> ETuple | None:
    min_exp: ETuple | None = None
    square_support = (hom_deg for hom_deg in d_csr.keys() if hom_deg + degree in d_csc)
    for hom_deg in square_support:
        exp = min_exp_in_product(d_csr[hom_deg + degree], d_csc[hom_deg])
        if exp is not None:
            if min_exp is None:
                min_exp = exp
            elif exp < min_exp:
                min_exp = exp

    return min_exp
