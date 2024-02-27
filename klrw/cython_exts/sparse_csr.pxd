from sage.rings.ideal import Ideal_generic as Ideal

from .sparse_csc cimport CSC_Mat

#cdef extern from "sparse_csc.pxd":
#    class Mat_CSC:
#        pass

#Making our version of CSR matrices, because scipy rejects working with object entries
cdef class CSR_Mat:
    cdef object[::1] data
    cdef int[::1] indices
    cdef int[::1] indptrs
    cdef int number_of_columns

    cpdef int nnz(self)
    cpdef CSC_Mat to_csc(self)
    cpdef bint is_zero(self)
    cpdef bint is_zero_mod(self, ideal : Ideal)
    
