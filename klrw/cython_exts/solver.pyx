import cython
import numpy as np
from cython.cimports import numpy as np

# from sys import intern

from scipy.sparse import csr_matrix, csc_matrix, spmatrix

from sage.rings.polynomial.polydict import ETuple
from sage.rings.integer_ring import ZZ

from klrw.klrw_algebra import KLRWAlgebra, KLRWElement

from cython.cimports.klrw.cython_exts.sparse_csr import CSR_Mat
from cython.cimports.klrw.cython_exts.sparse_csc import CSC_Mat
from .sparse_multiplication import multiply


@cython.cfunc
def mod_h_sq(
    exponents: ETuple, h_position: cython.int, u_position: cython.int, order: cython.int
) -> cython.bint:
    return exponents[h_position] == order


@cython.cfunc
def mod_u_sq_h(
    exponents: ETuple,
    h_position: cython.int,
    u_position: cython.int,
    order: cython.int,
) -> cython.bint:
    return exponents[u_position] == order and exponents[h_position] == 0


@cython.cclass
class Solver:
    KLRW: KLRWAlgebra
    _tolerance: cython.double
    verbose: cython.bint
    N: cython.int
    d0_csc: CSC_Mat
    d1_csc: CSC_Mat
    number_of_variables: cython.int
    h_position: cython.int
    u_position: cython.int
    h: object
    u: object
    u_multipliers: dict

    def __init__(self, KLRW, verbose=True):
        self.KLRW = KLRW
        self._tolerance = 0
        self.verbose = verbose
        self.N: cython.int = -1
        self.number_of_variables = -1

        quiver = self.KLRW.quiver
        V = quiver.non_framing_nodes()[0]
        W = V._replace(framing=True)
        self.h_position = self.KLRW.base().variables[V].position
        self.h = self.KLRW.base().variables[V].monomial
        self.u_position = self.KLRW.base().variables[V, W].position
        self.u = self.KLRW.base().variables[V, W].monomial
        self.u_multipliers = {V: self.u}
        print("h position: ", self.h_position, "u position: ", self.u_position)

    def KLRW_algebra(self):
        """
        Returns the KLRW algebra.
        """
        return self.KLRW

    def tolerance(self):
        return self._tolerance

    @cython.ccall
    def d0(self):
        return self.d0_csc

    def set_d0(self, d0_csc: CSC_Mat):
        # remember the number of T-branes if not known before
        if self.N != -1:
            assert self.N == d0_csc.number_of_rows, "Number of thimbles must match."
            assert (
                len(d0_csc.indptrs) == d0_csc.number_of_rows + 1
            ), "The differential matrix must be square."
        else:
            self.N = d0_csc.number_of_rows

        self.d0_csc = d0_csc

    def set_d1(self, d1_csc: CSC_Mat, number_of_variables):
        # remember the number of T-branes if not known before
        if self.N != -1:
            assert self.N == d1_csc.number_of_rows, "Number of thimbles must match."
            assert (
                len(d1_csc.indptrs) == d1_csc.number_of_rows + 1
            ), "The differential matrix must be square."
        else:
            self.N = d1_csc.number_of_rows

        self.d1_csc = d1_csc
        self.number_of_variables = number_of_variables

    def _d1_(self):
        return self.d1_csc

    # @cython.ccall
    # @cython.cfunc
    def make_system_for_corrections(
        self, multiplier: object, graded_type: str, order: cython.int = 1
    ):
        # d0_csc = self.d0_csc
        # d1_csc = self.d1_csc
        d0_csr: CSR_Mat = self.d0_csc.to_csr()
        d1_csr: CSR_Mat = self.d1_csc.to_csr()
        if self.verbose:
            print("Making the system")

        N: cython.int = self.N

        if graded_type == "h^order":
            condition = mod_h_sq
        elif graded_type == "u^order*h^0":
            condition = mod_u_sq_h
        else:
            raise ValueError("Unknown modulo type")

        A_csr_data_list: list = []
        A_csr_indices_list: list = []
        A_csr_indptrs_list: list = [0]
        b_data_list: list = []
        b_indices_list: list = []

        n: cython.int
        i: cython.int
        j: cython.int
        indptr1: cython.int
        indptr2: cython.int
        indptr1_end: cython.int
        indptr2_end: cython.int
        equations_so_far: cython.int = 0
        entries_so_far: cython.int = 0
        entry_can_be_non_zero: cython.bint = False

        ij_dict: dict
        for i in range(N):
            for j in range(N):
                # computing d0*d1 part
                # keep (i,j)-th entry as dictionary
                # {(KLRW_symmetric_dots_and_u_hbar,KLRWbraid): dict_representing_row}
                # (KLRW_dots_and_u_hbar,KLRWbraid) parametrizes rows in the system
                # dict_representing_row is a dictionary
                # {variable_number,system_coefficient}
                indptr1 = d0_csr.indptrs[i]
                indptr2 = self.d1_csc.indptrs[j]
                # indptr2 = d1_csc.indptrs[j]
                indptr1_end = d0_csr.indptrs[i + 1]
                indptr2_end = self.d1_csc.indptrs[j + 1]
                # indptr2_end = d1_csc.indptrs[j+1]
                while indptr1 != indptr1_end and indptr2 != indptr2_end:
                    # if d0_csr.indices[indptr1] == d1_csc.indices[indptr2]:
                    if d0_csr.indices[indptr1] == self.d1_csc.indices[indptr2]:
                        if not entry_can_be_non_zero:
                            ij_dict = {}
                            entry_can_be_non_zero = True

                        d0_element = d0_csr.data[indptr1]
                        # d1_elements = d1_csc.data[indptr2]
                        d1_elements = self.d1_csc.data[indptr2]
                        for n, d1_elem in d1_elements.items():
                            if n == -1:
                                print("+")
                            d1_element = multiplier * d1_elements[n]
                            d1_element = self.KLRW.scale_dots_in_element(
                                d1_element, self.u_multipliers
                            )
                            KLRW_element = d0_element * d1_element
                            for basis_vector, coef in KLRW_element:
                                for exp, scalar in coef.iterator_exp_coeff():
                                    if condition(
                                        exp, self.h_position, self.u_position, order
                                    ):
                                        if (exp, basis_vector) in ij_dict:
                                            row = ij_dict[exp, basis_vector]
                                            if n in row:
                                                row[n] += scalar
                                            else:
                                                row[n] = scalar
                                        else:
                                            ij_dict[exp, basis_vector] = {n: scalar}

                        indptr1 += 1
                        indptr2 += 1
                    elif d0_csr.indices[indptr1] < self.d1_csc.indices[indptr2]:
                        # elif d0_csr.indices[indptr1] < d1_csc.indices[indptr2]:
                        indptr1 += 1
                    else:
                        indptr2 += 1

                # computing d1*d0 part
                indptr1 = d1_csr.indptrs[i]
                # indptr2 = d0_csc.indptrs[j]
                indptr2 = self.d0_csc.indptrs[j]
                indptr1_end = d1_csr.indptrs[i + 1]
                # indptr2_end = d0_csc.indptrs[j+1]
                indptr2_end = self.d0_csc.indptrs[j + 1]
                while indptr1 != indptr1_end and indptr2 != indptr2_end:
                    # if d1_csr.indices[indptr1] == d0_csc.indices[indptr2]:
                    if d1_csr.indices[indptr1] == self.d0_csc.indices[indptr2]:
                        if not entry_can_be_non_zero:
                            ij_dict = {}
                            entry_can_be_non_zero = True

                        d1_elements = d1_csr.data[indptr1]
                        # d0_element = d0_csc.data[indptr2]
                        d0_element = self.d0_csc.data[indptr2]
                        for n in d1_elements:
                            if n == -1:
                                print("+")
                            d1_element = multiplier * d1_elements[n]
                            d1_element = self.KLRW.scale_dots_in_element(
                                d1_element, self.u_multipliers
                            )
                            KLRW_element = d1_element * d0_element
                            for basis_vector, coef in KLRW_element:
                                for exp, scalar in coef.iterator_exp_coeff():
                                    if condition(
                                        exp, self.h_position, self.u_position, order
                                    ):
                                        if (exp, basis_vector) in ij_dict:
                                            row = ij_dict[exp, basis_vector]
                                            if n in row:
                                                row[n] += scalar
                                            else:
                                                row[n] = scalar
                                        else:
                                            ij_dict[exp, basis_vector] = {n: scalar}

                        indptr1 += 1
                        indptr2 += 1
                    # elif d1_csr.indices[indptr1] < d0_csc.indices[indptr2]:
                    elif d1_csr.indices[indptr1] < self.d0_csc.indices[indptr2]:
                        indptr1 += 1
                    else:
                        indptr2 += 1

                # if the entry could be non-zero, computing -d0*d0 part
                # don't forget the sign inside!
                # using n=-1 for keeping in a row
                if entry_can_be_non_zero:
                    indptr1 = d0_csr.indptrs[i]
                    # indptr2 = d0_csc.indptrs[j]
                    indptr2 = self.d0_csc.indptrs[j]
                    indptr1_end = d0_csr.indptrs[i + 1]
                    # indptr2_end = d0_csc.indptrs[j+1]
                    indptr2_end = self.d0_csc.indptrs[j + 1]
                    while indptr1 != indptr1_end and indptr2 != indptr2_end:
                        # if d0_csr.indices[indptr1] == d0_csc.indices[indptr2]:
                        if d0_csr.indices[indptr1] == self.d0_csc.indices[indptr2]:
                            d0_element1 = d0_csr.data[indptr1]
                            # d0_element2 = d0_csc.data[indptr2]
                            d0_element2 = self.d0_csc.data[indptr2]
                            KLRW_element = d0_element1 * d0_element2
                            for basis_vector, coef in KLRW_element:
                                for exp, scalar in coef.iterator_exp_coeff():
                                    if condition(
                                        exp, self.h_position, self.u_position, order
                                    ):
                                        # here we keep only (exp,basis_vector)
                                        # in ij_dict,
                                        # the others should cancel in
                                        # a consistent system
                                        if (exp, basis_vector) in ij_dict:
                                            row = ij_dict[exp, basis_vector]
                                            if -1 in row:
                                                row[-1] += -scalar
                                            else:
                                                row[-1] = -scalar
                                        # else:
                                        #    raise ValueError("Inconsistent System")

                            indptr1 += 1
                            indptr2 += 1
                        # elif d0_csr.indices[indptr1] < d0_csc.indices[indptr2]:
                        elif d0_csr.indices[indptr1] < self.d0_csc.indices[indptr2]:
                            indptr1 += 1
                        else:
                            indptr2 += 1

                if entry_can_be_non_zero:
                    for __, row in ij_dict.items():
                        variables: list = list(row.keys())
                        variables.sort()
                        variable_number: cython.int
                        for variable_number in variables:
                            if variable_number == -1:
                                b_data_list.append(row[-1])
                                b_indices_list.append(equations_so_far)
                            else:
                                scalar: cython.int = row[variable_number]
                                A_csr_data_list.append(scalar)
                                A_csr_indices_list.append(variable_number)
                                entries_so_far += 1

                        A_csr_indptrs_list.append(entries_so_far)
                        equations_so_far += 1

                    entry_can_be_non_zero = False

        number_of_equations: cython.int = equations_so_far

        if self.verbose:
            print(
                "We have",
                number_of_equations,
                "rows",
                self.number_of_variables,
                "columns",
            )

        A_csr = csr_matrix(
            (A_csr_data_list, A_csr_indices_list, A_csr_indptrs_list),
            shape=(number_of_equations, self.number_of_variables),
        )

        b = csc_matrix(
            (b_data_list, b_indices_list, (0, len(b_indices_list))),
            shape=(number_of_equations, 1),
        )

        if self.verbose:
            print("Number of terms to correct:", len(b.data))

        return A_csr, b

    @cython.cfunc
    def solve_system_for_differential(
        self, M: spmatrix, bb: spmatrix, method="pypardiso"
    ):
        """
        Returns a tuple (solution : double[::1], is_integral : bool)
        """
        if self.verbose:
            print("Solving the system")

        # scipy conjugate gradients only take dense vectors, so we convert
        # A1 flattens array
        y = bb.todense().A1

        if method == "pypardiso":
            import pypardiso

            # pypardiso requires floats not integers
            M = M.asfptype()
            x = pypardiso.spsolve(M, y)

            if x.shape == ():
                x = x.reshape((1,))
        elif method == "cg":
            from scipy.sparse.linalg import cg

            x, exit_code = cg(M, y)
        else:
            raise ValueError("Unknown method")

        # trying an integer approximation if it works
        x_int = np.rint(x)

        assert np.allclose(M @ x_int, y), "Solution is not integral"

        return x_int

    @cython.cfunc
    def update_differential(self, x: cython.int[::1], multiplier: KLRWElement = 1):
        i: cython.int
        x_nnz: cython.int = 0
        for i in range(len(x)):
            if x[i] != 0:
                x_nnz += 1
        if self.verbose:
            print("Correcting {} matrix elements".format(x_nnz))

        number_of_columns: cython.int = self.N

        correted_csc_indptrs: cython.int[::1] = np.zeros(
            number_of_columns + 1, dtype=np.dtype("intc")
        )
        max_non_zero_entries: cython.int = self.d0_csc.nnz() + x_nnz
        correted_csc_indices: cython.int[::1] = np.zeros(
            max_non_zero_entries, dtype=np.dtype("intc")
        )
        correted_csc_data: object[::1] = np.empty(max_non_zero_entries, dtype=object)

        n: cython.int
        j: cython.int
        indptr1: cython.int
        indptr2: cython.int
        indptr1_end: cython.int
        indptr2_end: cython.int

        non_zero_entries_so_far: cython.int = 0

        for j in range(number_of_columns):
            indptr1 = self.d0_csc.indptrs[j]
            indptr2 = self.d1_csc.indptrs[j]
            indptr1_end = self.d0_csc.indptrs[j + 1]
            indptr2_end = self.d1_csc.indptrs[j + 1]
            while indptr1 != indptr1_end or indptr2 != indptr2_end:
                # TODO: better cases
                if indptr1 == indptr1_end:
                    entry = self.KLRW.zero()
                    for n, d1_entry in self.d1_csc.data[indptr2].items():
                        new_entry = multiplier * self.KLRW.base()(x[n]) * d1_entry
                        new_entry = self.KLRW.scale_dots_in_element(
                            new_entry, self.u_multipliers
                        )
                        entry += new_entry
                    if not entry.is_zero():
                        correted_csc_data[non_zero_entries_so_far] = entry
                        correted_csc_indices[non_zero_entries_so_far] = (
                            self.d1_csc.indices[indptr2]
                        )
                        non_zero_entries_so_far += 1

                    indptr2 += 1

                elif indptr2 == indptr2_end:
                    correted_csc_data[non_zero_entries_so_far] = self.d0_csc.data[
                        indptr1
                    ]
                    correted_csc_indices[non_zero_entries_so_far] = self.d0_csc.indices[
                        indptr1
                    ]
                    non_zero_entries_so_far += 1

                    indptr1 += 1

                elif self.d0_csc.indices[indptr1] == self.d1_csc.indices[indptr2]:
                    entry = self.d0_csc.data[indptr1]
                    for n, d1_entry in self.d1_csc.data[indptr2].items():
                        new_entry = multiplier * self.KLRW.base()(x[n]) * d1_entry
                        new_entry = self.KLRW.scale_dots_in_element(
                            new_entry, self.u_multipliers
                        )
                        entry += new_entry
                    if not entry.is_zero():
                        correted_csc_data[non_zero_entries_so_far] = entry
                        correted_csc_indices[non_zero_entries_so_far] = (
                            self.d0_csc.indices[indptr1]
                        )
                        non_zero_entries_so_far += 1

                    indptr1 += 1
                    indptr2 += 1

                elif self.d0_csc.indices[indptr1] < self.d1_csc.indices[indptr2]:
                    correted_csc_data[non_zero_entries_so_far] = self.d0_csc.data[
                        indptr1
                    ]
                    correted_csc_indices[non_zero_entries_so_far] = self.d0_csc.indices[
                        indptr1
                    ]
                    non_zero_entries_so_far += 1

                    indptr1 += 1

                else:
                    entry = self.KLRW.zero()
                    for n, d1_entry in self.d1_csc.data[indptr2].items():
                        new_entry = multiplier * self.KLRW.base()(x[n]) * d1_entry
                        new_entry = self.KLRW.scale_dots_in_element(
                            new_entry, self.u_multipliers
                        )
                        entry += new_entry
                    if not entry.is_zero():
                        correted_csc_data[non_zero_entries_so_far] = entry
                        correted_csc_indices[non_zero_entries_so_far] = (
                            self.d1_csc.indices[indptr2]
                        )
                        non_zero_entries_so_far += 1

                    indptr2 += 1

            correted_csc_indptrs[j + 1] = non_zero_entries_so_far

        # Deleting tails of None's in data and zeroes in indices
        correted_csc_indices = np.resize(
            correted_csc_indices, (non_zero_entries_so_far,)
        )
        correted_csc_data = np.resize(correted_csc_data, (non_zero_entries_so_far,))

        self.d0_csc = CSC_Mat(
            correted_csc_data,
            correted_csc_indices,
            correted_csc_indptrs,
            number_of_columns,
        )

    def make_corrections(
        self,
        multiplier=1,
        order: cython.int = 1,
        graded_type="h^order",
        method="pypardiso",
    ):
        """
        Solves the system (d0 + multiplier*d1)^2 = 0
        in a graded component denoted by graded_type and order.
        If graded_type = "h^order" the graded component
        is all terms with h^order [all powers of u possible]
        If graded_type = "u^order*h^0" the graded component
        is all terms with u^order*h^0
        """
        A_csr, b = self.make_system_for_corrections(
            multiplier=multiplier, order=order, graded_type=graded_type
        )

        A_csc = A_csr.tocsc()
        columns_to_remove = 0
        for i in range(len(A_csc.indptr) - 1):
            if A_csc.indptr[i] == A_csc.indptr[i + 1]:
                columns_to_remove += 1
                print("Zero column in the original matrix:", i)

        if columns_to_remove > 0:
            number_of_old_indices: cython.int = A_csc.shape[1]
            number_of_new_indices: cython.int = A_csc.shape[1] - columns_to_remove
            new_to_old_index = np.zeros(number_of_new_indices, dtype="intc")
            A_csc_indptrs_new = np.zeros(number_of_new_indices + 1, dtype="intc")
            new_indices_so_far: cython.int = 0
            i: cython.int
            for i in range(A_csc.shape[1]):
                # if we don't delete this column
                if A_csc.indptr[i] < A_csc.indptr[i + 1]:
                    new_to_old_index[new_indices_so_far] = i
                    A_csc_indptrs_new[new_indices_so_far + 1] = A_csc.indptr[i + 1]
                    new_indices_so_far += 1
            A = csc_matrix(
                (A_csc.data, A_csc.indices, A_csc_indptrs_new),
                shape=(A_csc.shape[0], number_of_new_indices),
            )
        else:
            A = A_csc

        from pickle import dump

        with open("matrix_local", "wb") as f:
            dump(A, file=f)
        with open("vector_local", "wb") as f:
            dump(b, file=f)

        if self.verbose:
            print("Transfoming to a symmetric square system")
        A_tr = A.transpose()
        M = A_tr @ A
        bb = A_tr @ b

        for i in range(len(M.indptr) - 1):
            if M.indptr[i] == M.indptr[i + 1]:
                print("Zero column in the square matrix:", i)

        x = self.solve_system_for_differential(M, bb, method=method)
        del M, bb

        if columns_to_remove > 0:
            x_modified = np.zeros(number_of_old_indices, dtype="intc")
            i: cython.int
            for i in range(number_of_new_indices):
                x_modified[new_to_old_index[i]] = x[i]
            x = x_modified
        else:
            x = x.astype(dtype="intc")

        # if still working over Z, comparison is exact
        if self.KLRW.scalars() == ZZ:
            assert np.array_equal(
                A_csc @ x.astype(np.dtype("intc")), b.todense().A1
            ), "Not a solution!"
        else:
            assert np.allclose(
                A_csc @ x, b.todense().A1, atol=self.tolerance()
            ), "Not a solution!"

        if self.verbose:
            print("Found a solution!")
        del A, b

        self.update_differential(x, multiplier)

    def check_d0(self):
        d0_squared_csr = multiply(self.d0_csc.to_csr(), self.d0_csc)
        print("d0 squares to zero:", d0_squared_csr.is_zero())

        hucenter = self.KLRW.base().ideal([self.u, self.h])
        print(
            "d0 squares to zero mod (u,h):",
            d0_squared_csr.is_zero_mod(hucenter),
        )

        husqcenter = self.KLRW.base().ideal([self.u**2, self.h])
        print(
            "d0 squares to zero mod (u^2,h):",
            d0_squared_csr.is_zero_mod(husqcenter),
        )

        hucucenter = self.KLRW.base().ideal([self.u**3, self.h])
        print(
            "d0 squares to zero mod (u^3,h):",
            d0_squared_csr.is_zero_mod(hucucenter),
        )

        hcenter = self.KLRW.base().ideal([self.h])
        print(
            "d0 squares to zero mod h:",
            d0_squared_csr.is_zero_mod(hcenter),
        )

        hsqcenter = self.KLRW.base().ideal([self.h**2])
        print(
            "d0 squares to zero mod h^2:",
            d0_squared_csr.is_zero_mod(hsqcenter),
        )

    def d0_squares_to_zero(self):
        d0_squared_csr = multiply(self.d0_csc.to_csr(), self.d0_csc)
        d0_squared_is_zero = d0_squared_csr.is_zero()
        print("d0 squares to zero:", d0_squared_is_zero)
        return d0_squared_is_zero

    def d0_squares_to_zero_mod(self, ideal):
        d0_squared_csr = multiply(self.d0_csc.to_csr(), self.d0_csc)
        d0_squared_is_zero = d0_squared_csr.is_zero_mod(ideal)
        print("d0 squares to zero modulo the ideal:", d0_squared_is_zero)
        return d0_squared_is_zero
