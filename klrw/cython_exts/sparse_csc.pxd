# from sparse_csr cimport CSR_Mat
from .sparse_csr cimport CSR_Mat

from sage.rings.ideal import Ideal_generic as Ideal

#Making our version of CSR matrices, because scipy rejects working with object entries
cdef class CSC_Mat:
    cdef object[::1] data
    cdef int[::1] indices
    cdef int[::1] indptrs
    cdef int number_of_rows

    cpdef int nnz(self)
    cpdef CSR_Mat to_csr(self)
    cpdef bint is_zero(self)
    cpdef bint is_zero_mod(self, ideal: Ideal)
