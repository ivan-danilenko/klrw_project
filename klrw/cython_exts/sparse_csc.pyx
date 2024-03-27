import cython
from cython.cimports import cython
import numpy as np
from cython.cimports import numpy as np

# from .sparse_csr cimport CSR_Mat
from scipy.sparse import csc_matrix

from klrw.klrw_algebra import KLRWElement


# Making our version of CSC matrices, because scipy rejects working with object entries
@cython.cclass
class CSC_Mat:
    #    data: object[::1]
    #    indices: cython.int[::1]
    #    indptrs: cython.int[::1]
    #    number_of_rows: cython.int

    def __init__(self, data, indices, indptrs, number_of_rows):
        assert len(data) == len(indices)
        assert indptrs[0] == 0
        assert indptrs[-1] == len(data), repr(indptrs[-1]) + " " + repr(len(data))
        # more checks?

        self.data = data
        self.indices = indices
        self.indptrs = indptrs
        self.number_of_rows = number_of_rows

    def __reduce__(self):
        return self.__class__, (
            np.asarray(self.data),
            np.asarray(self.indices),
            np.asarray(self.indptrs),
            self.number_of_rows,
        )

    def print_sizes(self):
        print(len(self.data), len(self.indices), len(self.indptrs), self.number_of_rows)

    def print_indices(self):
        print(np.asarray(self.indices))

    def print_indptrs(self):
        print(np.asarray(self.indptrs))

    def _data(self):
        return self.data

    def _indices(self):
        return self.indices

    def _indptrs(self):
        return self.indptrs

    def _number_of_rows(self):
        return self.number_of_rows

    def dict(self):
        i: cython.int
        j: cython.int

        d = {}

        for i in range(len(self.indptrs) - 1):
            for j in range(self.indptrs[i], self.indptrs[i + 1]):
                d[self.indices[j], i] = self.data[j]

        return d

    @cython.ccall
    def nnz(self) -> cython.int:
        return len(self.indices)

    # TODO: return KLRW.zero() if not present
    def __getitem__(self, key):
        """
        Returns (i,j)-th element if present
        Returns None is not present
        """
        assert len(key) == 2
        i: cython.int = key[0]
        j: cython.int = key[1]
        assert i >= 0
        assert i <= self.number_of_rows
        assert j >= 0
        assert j < len(self.indptrs)

        ind: cython.int = self.indptrs[j]
        ind_end: cython.int = self.indptrs[j + 1]
        while ind != ind_end:
            ii: cython.int = self.indices[ind]
            if ii < i:
                ind += 1
            elif ii == i:
                return self.data[ind]
            else:
                break
        return None

    @cython.ccall
    def to_csr(self) -> CSR_Mat:
        # scipy doesn't support matrix multiplication
        # and conversion to CSR with non-standard coefficients
        # Nevertheless we can use it to convert from one sparce matrix form to another
        # Since it's basically resorting indices + some compression.
        # Scipy will to do it fast
        M_csc = csc_matrix(
            (range(1, len(self.indices) + 1), self.indices, self.indptrs),
            shape=(self.number_of_rows, len(self.indptrs) - 1),
        )
        M_csr = M_csc.tocsr()
        del M_csc

        csr_data: object[::1] = np.empty(len(self.data), dtype=object)
        i: cython.int
        entry: cython.int
        for i in range(self.nnz()):
            entry = M_csr.data[i]
            csr_data[i] = self.data[entry - 1]
        csr_indices: cython.int[::1] = M_csr.indices.astype(dtype=np.dtype("intc"))
        csr_indptrs: cython.int[::1] = M_csr.indptr.astype(dtype=np.dtype("intc"))

        return CSR_Mat(csr_data, csr_indices, csr_indptrs, len(self.indptrs) - 1)

    @cython.ccall
    def is_zero(self) -> cython.bint:
        i: cython.int
        for i in range(self.nnz()):
            for x, coef in self.data[i]:
                if not coef.is_zero():
                    return False

        return True

    @cython.ccall
    def is_zero_mod(self, ideal: Ideal) -> cython.bint:
        i: cython.int
        for i in range(self.nnz()):
            for x, coef in self.data[i]:
                if not ideal.reduce(coef).is_zero():
                    return False

        return True

    def squares_to_zero(self, mod_ideal=None):
        csr = self.to_csr()

        assert self.number_of_rows == len(self.indptrs) - 1
        N: cython.int = self.number_of_rows

        i: cython.int
        j: cython.int
        indptr1: cython.int
        indptr2: cython.int
        indptr1_end: cython.int
        indptr2_end: cython.int

        entry_can_be_non_zero: cython.bint = False

        for i in range(N):
            for j in range(N):
                indptr1 = csr.indptrs[i]
                indptr2 = self.indptrs[j]
                indptr1_end = csr.indptrs[i + 1]
                indptr2_end = self.indptrs[j + 1]
                while indptr1 != indptr1_end and indptr2 != indptr2_end:
                    if csr.indices[indptr1] == self.indices[indptr2]:
                        if not entry_can_be_non_zero:
                            dot_product: KLRWElement = (
                                csr.data[indptr1] * self.data[indptr2]
                            )
                            entry_can_be_non_zero = True
                        else:
                            dot_product += csr.data[indptr1] * self.data[indptr2]
                        indptr1 += 1
                        indptr2 += 1
                    elif csr.indices[indptr1] < self.indices[indptr2]:
                        indptr1 += 1
                    else:
                        indptr2 += 1

                if entry_can_be_non_zero:
                    if mod_ideal is None:
                        if not dot_product.is_zero():
                            return False
                    else:
                        for _, coef in dot_product:
                            if mod_ideal.reduce(coef) != 0:
                                return False

        # if never seen a non-zero matrix element
        return True

    @classmethod
    def from_dict(cls, d_dict: dict, number_of_rows, number_of_columns):
        number_of_entries = len(d_dict)
        d_csc_data = np.empty(number_of_entries, dtype="O")
        d_csc_indices = np.zeros(number_of_entries, dtype="intc")
        d_csc_indptrs = np.zeros(number_of_columns + 1, dtype="intc")

        entries_so_far = 0
        current_j = 0
        for i, j in sorted(d_dict.keys(), key=lambda x: (x[1], x[0])):
            for a in range(current_j + 1, j + 1):
                d_csc_indptrs[a] = entries_so_far
            current_j = j
            # entries_so_far becomes the index of a new defomation variable
            d_csc_data[entries_so_far] = d_dict[i, j]
            d_csc_indices[entries_so_far] = i
            entries_so_far += 1
        for a in range(current_j + 1, number_of_columns + 1):
            d_csc_indptrs[a] = entries_so_far

        return cls(
            data=d_csc_data,
            indices=d_csc_indices,
            indptrs=d_csc_indptrs,
            number_of_rows=number_of_rows,
        )