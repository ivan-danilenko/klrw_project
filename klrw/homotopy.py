from typing import Iterable
import multiprocessing as mp
import numpy as np
from itertools import starmap

from scipy.sparse import block_array, dok_matrix

from sage.rings.integer_ring import ZZ

from klrw.klrw_algebra import KLRWAlgebra
from klrw.cython_exts.sparse_csc import CSC_Mat
from klrw.cython_exts.sparse_csr import CSR_Mat
from klrw.gradings import HomologicalGradingGroupElement
from klrw.perfect_complex import (
    KLRWDirectSumOfProjectives_Homomorphism,
    KLRWPerfectComplex,
    KLRWPerfectComplex_Ext0,
    ShiftedKLRWPerfectComplex,
)
from klrw.cython_exts.perfect_complex_corrections_ext import (
    CorrectionsMatrix,
    _etuple_lt_,
)
from klrw.cython_exts.homotopy_ext import (
    system_d_h_piece,
    system_h_d_piece,
    system_rhs_piece,
    make_homotopy_dict,
)


def chain_map_by_correction(
    initial_approximation_data,
    domain,
    codomain,
    max_iterations=10,
    parallel_processes=1,
    verbose=True,
    check=False,
):
    HomsetClass = domain.HomsetClassWithoutDifferential
    homset = HomsetClass(domain, codomain)
    initial_approximation = homset(initial_approximation_data)

    chain_map = homotopy(
        image_chain_map=None,
        initial_approximation=initial_approximation,
        max_iterations=max_iterations,
        parallel_processes=parallel_processes,
        verbose=verbose,
    )
    if chain_map is None:
        return None
    # do this instead of just `return chain_map`
    # to set the right parent: parent of chain maps,
    # not just graded maps.
    return domain.hom(codomain, chain_map, check=check, copy=False)


def homotopy(
    image_chain_map: KLRWDirectSumOfProjectives_Homomorphism | None,
    initial_approximation: KLRWDirectSumOfProjectives_Homomorphism | None = None,
    max_iterations=1,
    parallel_processes=1,
    verbose=True,
):
    """
    Retrun *a* solution `h` for `dh = g`, given chain map `g`.

    Input:
     -- `image_chain_map` is `g` in the notation below;
        if `image_chain_map is None`, we set `g = 0`.
     -- `initial_approximation` is `h_0` in the notation below.
        If `initial_approximation is None` then "h_0 = 0".
     -- `max_iterations` sets bound on how many iterations
        is allowed. If `max_iterations = 1`, everything is done in
        one step, no iteration. If `max_iterations > 1`,
        iterations are done. In the latter case, we assume that
        `dh_0` equals `g` modulo parameters, and parameters
        are not invertible (to avoid negative degrees).
    At least one of `g` and `h_0` must be given to find the domains
    and codomains of the maps.
    If a solution is not found, return `None`.

    There are two main settings:
     -- `g` is a chain map, so we construct a homotopy `h`;
     -- `g = 0`, then we construct a chain map `h`;
        without any further constraints we return `h = 0`.
        For this function to be meaningful,
        we should require that the geometric part of `h`,
        `h_0`, is given.

    We assume `g` is degree-preserving, so `h` is shifted.
    """
    if image_chain_map is None and initial_approximation is None:
        raise ValueError("`image_chain_map` or `initial_approximation` must be given")
    if image_chain_map is not None:
        domain = image_chain_map.domain()
        codomain = image_chain_map.codomain()
        shifted_codomain = codomain[-domain.differential.degree()]
    if initial_approximation is not None:
        if image_chain_map is None:
            domain = initial_approximation.domain()
            shifted_codomain = initial_approximation.codomain()
            codomain = shifted_codomain[domain.differential.degree()]
        else:
            assert domain == initial_approximation.domain()
            assert shifted_codomain == initial_approximation.codomain()

    data_homset = domain.hom_set(codomain)
    if image_chain_map is None:
        image_chain_map = data_homset.zero()
    if isinstance(image_chain_map, KLRWPerfectComplex_Ext0):
        image_chain_map = image_chain_map.representative()

    if image_chain_map.is_zero() and max_iterations == 1:
        HomsetClass = domain.HomsetClassWithoutDifferential
        solution_homset = HomsetClass(domain, shifted_codomain)
        return solution_homset.zero()
    assert (
        domain.KLRW_algebra().scalars() == ZZ
    ), "This solver works only over integers."
    diff_hom_degree = domain.differential.hom_degree()
    d_shifted_codomain_csc = {
        hom_deg: CSC_Mat.from_dict(
            mat.dict(copy=False),
            number_of_rows=shifted_codomain.component_rank(hom_deg + diff_hom_degree),
            number_of_columns=shifted_codomain.component_rank(hom_deg),
        )
        for hom_deg, mat in shifted_codomain.differential
    }

    d_domain_csr = {
        hom_deg: CSR_Mat.from_dict(
            mat.dict(copy=False),
            number_of_rows=domain.component_rank(hom_deg + diff_hom_degree),
            number_of_columns=domain.component_rank(hom_deg),
        )
        for hom_deg, mat in domain.differential
    }
    if max_iterations == 1:
        result = _homotopy_one_step_(
            image_chain_map=image_chain_map,
            domain=domain,
            codomain=codomain,
            shifted_codomain=shifted_codomain,
            d_domain_csr=d_domain_csr,
            d_shifted_codomain_csc=d_shifted_codomain_csc,
            parallel_processes=parallel_processes,
            verbose=verbose,
        )
    else:
        result = _homotopy_iterations_(
            image_chain_map=image_chain_map,
            initial_approximation=initial_approximation,
            domain=domain,
            codomain=codomain,
            shifted_codomain=shifted_codomain,
            d_domain_csr=d_domain_csr,
            d_shifted_codomain_csc=d_shifted_codomain_csc,
            parallel_processes=parallel_processes,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    return result


def _homotopy_iterations_(
    image_chain_map,
    initial_approximation,
    domain,
    codomain,
    shifted_codomain,
    d_domain_csr,
    d_shifted_codomain_csc,
    parallel_processes=1,
    max_iterations=10,
    verbose=True,
):
    klrw_algebra = domain.KLRW_algebra()
    end_algebra = klrw_algebra.opposite
    dot_algebra = klrw_algebra.base()
    assert (
        not dot_algebra.invertible_parameters
    ), "Invertible parameters are incompatible with iterations"

    def _geometric_part_(entry):
        return end_algebra(entry.value.geometric_part())

    d_domain_geom_csr = {
        hom_deg: mat.apply_entrywise(_geometric_part_, inplace=False).eliminate_zeros()
        for hom_deg, mat in d_domain_csr.items()
    }
    d_shifted_codomain_geom_csc = {
        hom_deg: mat.apply_entrywise(_geometric_part_, inplace=False).eliminate_zeros()
        for hom_deg, mat in d_shifted_codomain_csc.items()
    }

    if initial_approximation is not None:
        parent = initial_approximation.parent()
        new_image = image_chain_map - parent.apply_differential(initial_approximation)
    else:
        new_image = image_chain_map
    exp = min_exp_in_graded_map(new_image)
    if exp is None:
        if verbose:
            print("Solution found!")
        return initial_approximation
    exp = dot_algebra.etuple_ignoring_dots(exp)
    # print(dot_algebra.monomial(*exp))
    # print(new_image)
    if not initial_approximation.is_zero():
        assert (
            not exp.is_constant()
        ), "The initial chain map does not commute mod extra variables."

    approximation = initial_approximation
    for k in range(max_iterations):
        correction = _homotopy_one_step_(
            image_chain_map=new_image,
            domain=domain,
            codomain=codomain,
            shifted_codomain=shifted_codomain,
            d_domain_csr=d_domain_geom_csr,
            d_shifted_codomain_csc=d_shifted_codomain_geom_csc,
            relevant_parameter_part=exp,
            parallel_processes=parallel_processes,
            verbose=verbose,
        )
        if correction is None:
            return None

        if approximation is not None:
            approximation += correction
        else:
            approximation = correction
        parent = approximation.parent()
        new_image = image_chain_map - parent.apply_differential(approximation)
        exp = min_exp_in_graded_map(new_image)
        if exp is None:
            return approximation
        exp = dot_algebra.etuple_ignoring_dots(exp)
        # print(dot_algebra.monomial(*exp))
        # print(new_image)

    return None


def _homotopy_one_step_(
    image_chain_map,
    domain,
    codomain,
    shifted_codomain,
    d_domain_csr,
    d_shifted_codomain_csc,
    relevant_parameter_part=None,
    parallel_processes=1,
    verbose=True,
):
    terms = possible_terms(
        domain=domain,
        shifted_codomain=shifted_codomain,
        relevant_parameter_part=relevant_parameter_part,
        parallel_processes=parallel_processes,
        verbose=verbose,
    )
    if not terms:
        # inconsistent system
        return None

    # order homological degrees of possible homotopy terms
    ordered_term_degrees = order_hom_gradings(
        terms.keys(),
        domain.differential.hom_degree(),
        verbose=verbose,
    )
    # order homological degrees of possible chain maps
    chain_hom_degrees = (
        hom_deg for hom_deg in domain.gradings() if hom_deg in codomain.gradings()
    )
    ordered_chain_degrees = order_hom_gradings(
        chain_hom_degrees,
        domain.differential.hom_degree(),
        verbose=verbose,
    )

    basis_in_product = {}
    lhs = system_on_homotopy_lhs(
        klrw_algebra=domain.KLRW_algebra(),
        d_domain_csr=d_domain_csr,
        d_shifted_codomain_csc=d_shifted_codomain_csc,
        possible_terms=terms,
        ordered_term_degrees=ordered_term_degrees,
        ordered_chain_degrees=ordered_chain_degrees,
        differential_hom_degree=domain.differential.hom_degree(),
        basis_appearing_in_product=basis_in_product,
        relevant_parameter_part=relevant_parameter_part,
        verbose=verbose,
    )

    rhs = system_on_homotopy_rhs(
        image_chain_map=image_chain_map,
        ordered_chain_degrees=ordered_chain_degrees,
        basis_appearing_in_product=basis_in_product,
        relevant_parameter_part=relevant_parameter_part,
        verbose=verbose,
    )

    x = solve_system(lhs, rhs, verbose=verbose)

    homotopy_dict = make_homotopy_dict(
        klrw_algebra=domain.KLRW_algebra(),
        h_csc=terms,
        x_csc=x,
        ordered_term_degrees=ordered_term_degrees,
    )

    HomsetClass = domain.HomsetClassWithoutDifferential
    solution_homset = HomsetClass(domain, shifted_codomain)
    return solution_homset(homotopy_dict)


'''
def homotopy(
    image_chain_map: KLRWDirectSumOfProjectives_Homomorphism,
    parallel_processes=1,
    verbose=True,
):
    """
    Constuct a homotopy associated with `image_chain_map`.

    If `image_chain_map` is nil-homotopic, return a homotopy,
    i.e. a map `h`, such that
    `image_chain_map = d*h + h*d`.
    If `image_chain_map` is not nil-homotopic, return `None`.
    """
    if isinstance(image_chain_map, KLRWPerfectComplex_Ext0):
        image_chain_map = image_chain_map.representative()
    domain = image_chain_map.domain()
    codomain = image_chain_map.codomain()
    shifted_codomain = codomain[-domain.differential.degree()]
    HomsetClass = domain.HomsetClassWithoutDifferential
    homset = HomsetClass(domain, shifted_codomain)
    if image_chain_map.is_zero():
        return homset.zero()
    assert (
        domain.KLRW_algebra().scalars() == ZZ
    ), "This solver works only over integers."
    diff_hom_degree = domain.differential.hom_degree()
    d_shifted_codomain_csc = {
        hom_deg: CSC_Mat.from_dict(
            mat.dict(copy=False),
            number_of_rows=shifted_codomain.component_rank(hom_deg + diff_hom_degree),
            number_of_columns=shifted_codomain.component_rank(hom_deg),
        )
        for hom_deg, mat in shifted_codomain.differential
    }

    d_domain_csr = {
        hom_deg: CSR_Mat.from_dict(
            mat.dict(copy=False),
            number_of_rows=domain.component_rank(hom_deg + diff_hom_degree),
            number_of_columns=domain.component_rank(hom_deg),
        )
        for hom_deg, mat in domain.differential
    }

    terms = possible_terms(
        domain=domain,
        shifted_codomain=shifted_codomain,
        differential_degree=domain.differential.degree(),
        parallel_processes=parallel_processes,
        verbose=verbose,
    )
    if not terms:
        # inconsistent system
        return None

    # order homological degrees of possible homotopy terms
    ordered_term_degrees = order_hom_gradings(
        terms.keys(),
        domain.differential.hom_degree(),
        verbose=verbose,
    )
    # order homological degrees of possible chain maps
    chain_hom_degrees = (
        hom_deg for hom_deg in domain.gradings() if hom_deg in codomain.gradings()
    )
    ordered_chain_degrees = order_hom_gradings(
        chain_hom_degrees,
        domain.differential.hom_degree(),
        verbose=verbose,
    )

    basis_in_product = {}
    lhs = system_on_homotopy_lhs(
        klrw_algebra=domain.KLRW_algebra(),
        d_domain_csr=d_domain_csr,
        d_shifted_codomain_csc=d_shifted_codomain_csc,
        possible_terms=terms,
        ordered_term_degrees=ordered_term_degrees,
        ordered_chain_degrees=ordered_chain_degrees,
        differential_hom_degree=domain.differential.hom_degree(),
        basis_appearing_in_product=basis_in_product,
        verbose=verbose,
    )

    rhs = system_on_homotopy_rhs(
        image_chain_map=image_chain_map,
        ordered_chain_degrees=ordered_chain_degrees,
        basis_appearing_in_product=basis_in_product,
        verbose=verbose,
    )

    x = solve_system(lhs, rhs, verbose=verbose)
    # if the chain map is not nil-homotopic
    if x is None:
        return None

    homotopy_dict = make_homotopy_dict(
        domain=domain,
        shifted_codomain=shifted_codomain,
        h_csc=terms,
        x_csc=x,
        ordered_term_degrees=ordered_term_degrees,
    )

    return homset(homotopy_dict)
'''


def possible_terms(
    domain: KLRWPerfectComplex | ShiftedKLRWPerfectComplex,
    shifted_codomain: KLRWPerfectComplex | ShiftedKLRWPerfectComplex,
    parallel_processes: int,
    relevant_parameter_part=None,
    verbose=True,
):
    # terms of differential exist only between adjacent coh degrees
    diff_hom_degree = domain.differential.hom_degree()
    possible_terms_hom_degrees = frozenset(
        hom_deg
        for hom_deg in domain.gradings()
        if hom_deg in shifted_codomain.gradings()
    )
    klrw_algebra = domain.KLRW_algebra()
    tasks = [
        (
            klrw_algebra,
            domain.projectives(hom_deg),
            shifted_codomain.projectives(hom_deg),
            hom_deg,
            diff_hom_degree,
            relevant_parameter_part,
            verbose,
        )
        for hom_deg in possible_terms_hom_degrees
    ]
    if parallel_processes == 1:
        terms = dict(starmap(possible_terms_in_hom_degree_and_print, tasks))
    else:
        mp_context = mp.get_context("spawn")
        with mp_context.Pool(processes=parallel_processes) as pool:
            corrections_async_list = pool.starmap_async(
                possible_terms_in_hom_degree_and_print, tasks
            )
            terms = dict(corrections_async_list.get())

    return terms


def possible_terms_in_hom_degree_and_print(
    klrw_algebra: KLRWAlgebra,
    projectives_domain: list,
    projectives_codomain: list,
    hom_deg: HomologicalGradingGroupElement,
    diff_hom_degree: HomologicalGradingGroupElement,
    relevant_parameter_part=None,
    verbose=True,
):
    if verbose:
        print(
            "Start: Finding possible terms for A_{} -> B_{}".format(
                hom_deg, hom_deg - diff_hom_degree
            )
        )

    terms_in_hom_deg = possible_terms_in_hom_degree(
        klrw_algebra=klrw_algebra,
        projectives_domain=projectives_domain,
        projectives_codomain=projectives_codomain,
        relevant_parameter_part=relevant_parameter_part,
    )

    if verbose:
        print(
            "End: Finding possible terms for A_{} -> B_{}".format(
                hom_deg, hom_deg - diff_hom_degree
            )
        )

    return hom_deg, terms_in_hom_deg


def possible_terms_in_hom_degree(
    klrw_algebra: KLRWAlgebra,
    projectives_domain: list,
    projectives_codomain: list,
    relevant_parameter_part=None,
):
    # To record all possible terms we have a data structure similar
    # to CSR matrices.
    # indptrs and indices are the same as in CSR matrices:
    # the i-th row is [h_csc_indptrs[i]:h_csc_indptrs[i+1]]
    # in h_csc_indices and h_csc_entry_ptrs.
    # Let a be in range(h_csc_indptrs[i], h_csc_indptrs[i+1])
    # Then h_csc_indices[a] keeps the number of column of an entry;
    # Possible terms are in listed in
    # h_csc_corrections[h_csc_entry_ptrs[a]:h_csc_entry_ptrs[a+1]]
    #
    # At first we make lists, then use numpy arrays.
    from collections import deque

    h_csc_corrections_braid_deque = deque()  # []
    h_csc_corrections_exp_deque = deque()  # []
    h_csc_entry_ptrs_deque = deque([0])  # [0]
    h_csc_indices_deque = deque()  # []
    h_csc_indptrs = np.zeros(len(projectives_domain) + 1, dtype=np.dtype("intc"))
    entry_index = 0
    variable_index = 0
    # left_index is column index
    # right_index is row index
    for left_index, left_projective in enumerate(projectives_domain):
        for right_index, right_projective in enumerate(projectives_codomain):
            equ_degree = (
                right_projective.equivariant_degree - left_projective.equivariant_degree
            )
            graded_component = klrw_algebra[
                left_projective.state : right_projective.state : equ_degree
            ]
            basis = graded_component.basis(
                as_tuples=True,
                relevant_parameter_part=relevant_parameter_part,
            )

            if not basis:
                continue

            for braid, exp in basis:
                h_csc_corrections_braid_deque.append(braid)
                h_csc_corrections_exp_deque.append(exp)
                variable_index += 1
            h_csc_entry_ptrs_deque.append(variable_index)
            h_csc_indices_deque.append(right_index)
            entry_index += 1

        h_csc_indptrs[left_index + 1] = entry_index

    h_csc_corrections_braid = np.array(
        h_csc_corrections_braid_deque, dtype=np.dtype("O")
    )
    # to avoid making a 2d array
    h_csc_corrections_exp = np.empty(
        len(h_csc_corrections_exp_deque), dtype=np.dtype("O")
    )
    for i, val in enumerate(h_csc_corrections_exp_deque):
        h_csc_corrections_exp[i] = val
    h_csc_entry_ptrs = np.array(h_csc_entry_ptrs_deque, dtype=np.dtype("intc"))
    h_csc_indices = np.array(h_csc_indices_deque, dtype=np.dtype("intc"))

    return CorrectionsMatrix(
        h_csc_corrections_braid,
        h_csc_corrections_exp,
        h_csc_entry_ptrs,
        h_csc_indices,
        h_csc_indptrs,
        len(projectives_codomain),
    )


def order_hom_gradings(
    hom_degrees: Iterable,
    differential_hom_degree: HomologicalGradingGroupElement,
    verbose=True,
):
    # To have a good block-band matrix
    # `hom_deg - diff_hom_degree`
    # has to be right before or right after `hom_deg`.
    # For rank one (homological) gradings and
    # differential of degree +/-1 we can just sort.
    try:
        homological_grading_label = (
            differential_hom_degree.parent().homological_grading_label
        )
        d_hom_deg_int = differential_hom_degree.coefficient(homological_grading_label)
        assert d_hom_deg_int == 1 or d_hom_deg_int == -1
        return sorted(
            hom_degrees, key=lambda x: x.coefficient(homological_grading_label)
        )
    except (AttributeError, AssertionError):
        if verbose:
            print("Warning: current method does not make the system block-diagonal")
        return list(hom_degrees)


def system_on_homotopy_lhs(
    klrw_algebra: KLRWAlgebra,
    d_domain_csr: CSR_Mat,
    d_shifted_codomain_csc: CSC_Mat,
    possible_terms: dict[CorrectionsMatrix],
    ordered_term_degrees: list[HomologicalGradingGroupElement],
    ordered_chain_degrees: list[HomologicalGradingGroupElement],
    differential_hom_degree: HomologicalGradingGroupElement,
    basis_appearing_in_product: dict,
    relevant_parameter_part=None,
    verbose=True,
):
    matrix_blocks = [[None] * len(ordered_term_degrees) for _ in ordered_chain_degrees]

    term_hom_deg_to_index = {
        hom_deg: index for index, hom_deg in enumerate(ordered_term_degrees)
    }

    for row_hom_deg_index, hom_deg in enumerate(ordered_chain_degrees):
        if verbose:
            print(
                "Making the left hand side for the piece A_{} -> B_{}".format(
                    hom_deg, hom_deg
                )
            )

        # indices of homological gradings
        # `hom_deg` and `hom_deg + differential_hom_degree`
        # we keep them `None` if there are no possible homotopy terms
        # for that homological degree or if differentials are zero.
        col_indices = [None, None]
        if hom_deg in possible_terms and hom_deg in d_shifted_codomain_csc:
            col_indices[0] = term_hom_deg_to_index[hom_deg]
        if (
            hom_deg + differential_hom_degree in possible_terms
            and hom_deg in d_domain_csr
        ):
            col_indices[1] = term_hom_deg_to_index[hom_deg + differential_hom_degree]
        basis_appearing_in_product[hom_deg] = {}
        if col_indices[0] is not None:
            matrix_blocks[row_hom_deg_index][col_indices[0]] = system_d_h_piece(
                klrw_algebra=klrw_algebra,
                d_shifted_codomain_csc=d_shifted_codomain_csc[hom_deg],
                h_csc=possible_terms[hom_deg],
                basis_appearing_in_product=basis_appearing_in_product[hom_deg],
                relevant_parameter_part=relevant_parameter_part,
            )
        if col_indices[1] is not None:
            matrix_blocks[row_hom_deg_index][col_indices[1]] = system_h_d_piece(
                klrw_algebra=klrw_algebra,
                d_domain_csr=d_domain_csr[hom_deg],
                h_csc=possible_terms[hom_deg + differential_hom_degree],
                basis_appearing_in_product=basis_appearing_in_product[hom_deg],
                relevant_parameter_part=relevant_parameter_part,
            )
            if col_indices[0] is not None:
                A = matrix_blocks[row_hom_deg_index][col_indices[0]]
                B = matrix_blocks[row_hom_deg_index][col_indices[1]]
                A.resize(B.shape[0], A.shape[1])

    A = block_array(matrix_blocks, format="csr")

    return A


def system_on_homotopy_rhs(
    image_chain_map: KLRWDirectSumOfProjectives_Homomorphism,
    ordered_chain_degrees: list[HomologicalGradingGroupElement],
    basis_appearing_in_product: dict,
    relevant_parameter_part=None,
    verbose=True,
):
    augment_blocks = [[None] for _ in ordered_chain_degrees]
    for hom_deg, row in zip(ordered_chain_degrees, augment_blocks):
        basis_dict = basis_appearing_in_product[hom_deg]
        if hom_deg in image_chain_map.support():
            if verbose:
                print(
                    "Making the right hand side for the piece A_{} -> B_{}".format(
                        hom_deg, hom_deg
                    )
                )
            # rows have length 1
            row[0] = system_rhs_piece(
                image_chain_map=image_chain_map(hom_deg),
                basis_appearing_in_product=basis_dict,
                dot_algebra=image_chain_map.parent().KLRW_algebra().base(),
                relevant_parameter_part=relevant_parameter_part,
            )
        else:
            # rows have length 1
            row[0] = dok_matrix(
                (len(basis_dict), 1),
                dtype="intc",
            )

    b = block_array(augment_blocks, format="csc")
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
            # if model is infeasible,
            # i.e. the chain map is not nil-homotopic
            if m.Status == 3:
                return None
            x = x.X

    from scipy.sparse import csr_matrix

    # making a sparse integer-valued vector from x
    # we make it sparse because most entries are expected to be zero
    x = csr_matrix(x, dtype="intc")
    x = x.transpose()

    assert (A @ x - b).nnz == 0, "Integer approximation is not a solution!"

    return x


def min_exp_in_graded_map(graded_map):
    min_exp = None
    for hom_deg in graded_map.support():
        for entry in graded_map(hom_deg).dict(copy=False).values():
            for _, poly in entry.value:
                for exp in poly.exponents():
                    if min_exp is None:
                        min_exp = exp
                    elif _etuple_lt_(exp, min_exp):
                        min_exp = exp

    return min_exp
