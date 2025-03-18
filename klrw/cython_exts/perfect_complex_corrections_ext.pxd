cimport cython

cdef class CorrectionsMatrix:
    cdef object[::1] corrections_braid
    cdef object[::1] corrections_exp
    cdef cython.int[::1] entry_ptrs
    cdef cython.int[::1] indices
    cdef cython.int[::1] indptrs
    cdef cython.int number_of_rows

    cpdef _corrections_braid(self)

    cpdef _corrections_exp(self)
    cpdef _entry_ptrs(self)
    cpdef _indices(self)
    cpdef _indptrs(self)
    cpdef _number_of_columns(self)
    cpdef _number_of_rows(self)
