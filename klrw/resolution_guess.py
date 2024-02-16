from itertools import chain, product
from sage.matrix.constructor import matrix
from sage.rings.rational_field import RationalField


def maps_factoring_through_the_kernel(KLRW, C1, C2, d, projective):
    basis_iterator_C1 = (
        (i, basis)
        for i in range(len(C1))
        for basis in KLRW.basis_by_states_and_degrees(
            projective[0], C1[i][0], C1[i][1] - projective[1]
        )
    )
    basis_iterator_C2 = (
        (i, basis)
        for i in range(len(C2))
        for basis in KLRW.basis_by_states_and_degrees(
            projective[0], C2[i][0], C2[i][1] - projective[1]
        )
    )

    basis_index_C1 = {
        (state_index,) + basis_tuple: index
        for index, (state_index, basis_tuple) in enumerate(basis_iterator_C1)
    }
    basis_index_C2 = {
        (state_index,) + basis_tuple: index
        for index, (state_index, basis_tuple) in enumerate(basis_iterator_C2)
    }

    mm = matrix(RationalField(), len(basis_index_C1), len(basis_index_C2), sparse=True)

    for (i, braid, exp), column_ind in basis_index_C2.items():
        for j in range(len(C1)):
            for br, poly in KLRW.base().monomial(*exp) * KLRW.monomial(braid) * d[j, i]:
                for ex, coeff in poly.iterator_exp_coeff():
                    row_ind = basis_index_C1[j, br, ex]
                    mm[row_ind, column_ind] += coeff

    mat = matrix(KLRW, len(C2), mm.right_kernel().dimension(), sparse=True)
    for j, vector in enumerate(mm.right_kernel().basis()):
        for (i, braid, exp), ind in basis_index_C2.items():
            mat[i, j] += vector[ind] * KLRW.base().monomial(*exp) * KLRW.monomial(braid)

    for (i, j), elem in mat.dict(copy=False).items():
        assert elem.degree(check_if_homogeneous=True) == C2[i][1] - projective[1], (
            repr(i) + repr(j) + repr(elem)
        )
    assert (mat.transpose() * d.transpose()).is_zero()

    return mat


def filter_independent(KLRW, C2, C3, d, projective, new_d):
    basis_iterator_C2 = (
        (i, basis)
        for i in range(len(C2))
        for basis in KLRW.basis_by_states_and_degrees(
            projective[0], C2[i][0], C2[i][1] - projective[1]
        )
    )

    basis_iterator_C3 = (
        (i, basis)
        for i in range(len(C3))
        for basis in KLRW.basis_by_states_and_degrees(
            projective[0], C3[i][0], C3[i][1] - projective[1]
        )
    )

    basis_index_C2 = {
        (state_index,) + basis_tuple: index
        for index, (state_index, basis_tuple) in enumerate(basis_iterator_C2)
    }

    basis_index_C3 = {
        (state_index,) + basis_tuple: index
        for index, (state_index, basis_tuple) in enumerate(basis_iterator_C3)
    }

    # for x in basis_index_C2:
    #    print(x)
    # print("---")
    # for x in basis_index_C3:
    #    print(x)

    m = matrix(RationalField(), len(basis_index_C2), len(basis_index_C3), sparse=True)

    # print("++")
    for (i, braid, exp), column_ind in basis_index_C3.items():
        for j in range(len(C2)):
            # print((i, braid, exp), column_ind, j)
            # print(projective, C2[j], C3[i])
            # x = KLRW.base().monomial(*exp) * KLRW.monomial(braid)
            # y = d[j, i]
            # print(x, y)
            # print(x.degree(), y.degree(), (x*y).degree() )
            for br, poly in KLRW.base().monomial(*exp) * KLRW.monomial(braid) * d[j, i]:
                for ex, coeff in poly.iterator_exp_coeff():
                    # print(coeff)
                    row_ind = basis_index_C2[j, br, ex]
                    m[row_ind, column_ind] += coeff

    #    print("+++")
    n = matrix(RationalField(), len(basis_index_C2), new_d.ncols(), sparse=True)
    for (i, column_ind), item in new_d.dict(copy=False).items():
        for braid, poly in item:
            for exp, coeff in poly.iterator_exp_coeff():
                row_ind = basis_index_C2[i, braid, exp]
                n[row_ind, column_ind] += coeff

    #    print("1",m)
    #    print("2",n)
    rref = m.augment(n).rref()
    #    print("3",rref)
    #    rref = rref[-new_d.ncols():,:]
    #    print("4",rref)
    independent_indices = rref.pivots()
    new_independent_indices = (
        i - m.ncols() for i in independent_indices if i >= m.ncols()
    )

    #    print(rref.nrows(), len(independent_indices))

    #    print("++++")
    mat = new_d.matrix_from_columns(new_independent_indices)
    # matrix(KLRW, len(C2), len(new_independent_indices), sparse=True)
    # for j, index in enumerate(new_independent_indices):
    #    for ind, entry in new_d[index,:]:
    #        print(entry)
    #        mat[i, j] += rref[index,ind] * KLRW.base().monomial(*exp) *
    # KLRW.monomial(braid)

    return mat


def add_kernel_elements(
    KLRW,
    C1,
    C2,
    d1,
    C3_partial=None,
    d2_partial=None,
    filter=True,
    first_checks=tuple(),
    lowest_degree=-10,
    highest_degree=5,
):
    if C3_partial is None:
        C3_partial = []
    if d2_partial is None:
        d2_partial = matrix(KLRW, len(C2), 0, sparse=True)
    projectives_to_check = chain(
        first_checks,
        product(KLRW.state_set(), range(highest_degree, lowest_degree, -1)),
    )

    count = 0
    for state, i in projectives_to_check:
        projective = state, i
        dd = maps_factoring_through_the_kernel(KLRW, C1, C2, d1, projective)
        if not dd.is_zero():
            #            print(projective)
            # if len(C3)>0:
            if filter:
                dd = filter_independent(
                    KLRW, C2, C3_partial, d2_partial, projective, dd
                )
            #            print(dd)
            C3_partial += [projective] * dd.ncols()
            if dd.ncols() > 0:
                count += dd.ncols()
                print("Adding ", dd.ncols(), " elements for ", projective)
            d2_partial = d2_partial.augment(dd)
    print("In total ", count, " elements added.")
    return C3_partial, d2_partial
