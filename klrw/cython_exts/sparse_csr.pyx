import cython
from cython.cimports import cython
import numpy as np
from cython.cimports import numpy as np
from typing import Self

from scipy.sparse import csr_matrix

from klrw.klrw_algebra import KLRWElement

from .sparse_csc cimport CSC_Mat


# Making our version of CSR matrices, because scipy rejects working with object entries
@cython.cclass
class CSR_Mat:
    data: object[::1]
    indices: cython.int[::1]
    indptrs: cython.int[::1]
    number_of_columns: cython.int

    def __init__(self, data, indices, indptrs, number_of_columns):
        assert len(data) == len(indices)
        assert indptrs[0] == 0
        assert indptrs[-1] == len(data)
        # more checks?

        self.data = data
        self.indices = indices
        self.indptrs = indptrs
        self.number_of_columns = number_of_columns

    def _data(self):
        return self.data

    def _indices(self):
        return self.indices

    def _indptrs(self):
        return self.indptrs

    def _number_of_columns(self):
        return self.number_of_columns

    @cython.ccall
    def nnz(self) -> cython.int:
        return len(self.indices)

    def print_sizes(self):
        print(
            len(self.data), len(self.indices), len(self.indptrs), self.number_of_columns
        )

    def print_indices(self):
        print(np.asarray(self.indices))

    def print_indptrs(self):
        print(np.asarray(self.indptrs))

    @cython.ccall
    def to_csc(self) -> CSC_Mat:
        # scipy doesn't support matrix multiplication
        # and conversion to CSC with non-standard coefficients
        # Nevertheless we can use it to convert from one sparce matrix form to another
        # Since it's basically resorting indices + some compression.
        # Scipy will to do it fast
        M_csr = csr_matrix(
            (range(1, len(self.indices) + 1), self.indices, self.indptrs),
            shape=(len(self.indptrs) - 1, self.number_of_columns),
        )
        M_csc = M_csr.tocsr()
        del M_csr

        csc_data: object[::1] = np.empty(len(self.data), dtype=object)
        i: cython.int
        entry: cython.int
        for i in range(self.nnz()):
            entry = M_csc.data[i]
            csc_data[i] = self.data[entry - 1]
        csc_indices: cython.int[::1] = M_csc.indices.astype(dtype=np.dtype("intc"))
        csc_indptrs: cython.int[::1] = M_csc.indptr.astype(dtype=np.dtype("intc"))

        return CSC_Mat(csc_data, csc_indices, csc_indptrs, len(self.indptrs) - 1)

    @cython.ccall
    def is_zero(self: Self) -> cython.bint:
        i: cython.int
        for i in range(self.nnz()):
            for x, coef in self.data[i]:
                if not coef.is_zero():
                    return False

        return True

    @cython.ccall
    def is_zero_mod(self: Self, ideal: Ideal) -> cython.bint:
        i: cython.int
        for i in range(self.nnz()):
            for x, coef in self.data[i]:
                if not ideal.reduce(coef).is_zero():
                    return False

        return True
