import cython
from cython.cimports import cython
import numpy as np
from cython.cimports import numpy as np

from .sparse_csc cimport CSC_Mat
from .sparse_csr cimport CSR_Mat

# might give a time increase in speed,
# but will not give errors and indexing from the tail
@cython.boundscheck(False)
@cython.wraparound(False)
def multiply(
    M_csr: CSR_Mat,
    N_csc: CSC_Mat,
) -> CSR_Mat:

    M_number_of_rows: cython.int = len(M_csr.indptrs) - 1
    N_number_of_columns: cython.int = len(N_csc.indptrs) - 1

    product_csr_indptrs: cython.int[::1] = np.zeros(
        M_number_of_rows + 1, dtype=np.dtype("intc")
    )
    max_non_zero_entries: cython.int = len(M_csr.indices) * len(N_csc.indices)
    product_csr_indices: cython.int[::1] = np.zeros(
        max_non_zero_entries, dtype=np.dtype("intc")
    )
    product_csr_data: object[::1] = np.empty(max_non_zero_entries, dtype=object)

    i: cython.int
    j: cython.int
    indptr1: cython.int
    indptr2: cython.int
    indptr1_end: cython.int
    indptr2_end: cython.int

    non_zero_entries_so_far: cython.int = 0
    entry_can_be_non_zero: cython.bint = False

    for i in range(M_number_of_rows):
        for j in range(N_number_of_columns):
            indptr1 = M_csr.indptrs[i]
            indptr2 = N_csc.indptrs[j]
            indptr1_end = M_csr.indptrs[i + 1]
            indptr2_end = N_csc.indptrs[j + 1]
            while indptr1 != indptr1_end and indptr2 != indptr2_end:
                if M_csr.indices[indptr1] == N_csc.indices[indptr2]:
                    if not entry_can_be_non_zero:
                        dot_product = M_csr.data[indptr1] * N_csc.data[indptr2]
                        entry_can_be_non_zero = True
                    else:
                        dot_product += M_csr.data[indptr1] * N_csc.data[indptr2]
                    indptr1 += 1
                    indptr2 += 1
                elif M_csr.indices[indptr1] < N_csc.indices[indptr2]:
                    indptr1 += 1
                else:
                    indptr2 += 1

            if entry_can_be_non_zero:
                if not dot_product.is_zero():
                    product_csr_data[non_zero_entries_so_far] = dot_product
                    product_csr_indices[non_zero_entries_so_far] = j
                    non_zero_entries_so_far += 1
                entry_can_be_non_zero = False

        product_csr_indptrs[i + 1] = non_zero_entries_so_far

    # Deleting tails of None's in data and zeroes in indices
    # product_csr_indices = np.resize(product_csr_indices,(non_zero_entries_so_far,))
    # product_csr_data = np.resize(product_csr_data,(non_zero_entries_so_far,))
    # Deleting tails of None's in data and zeroes in indices
    # Since we don't want to recreate the array, we use a slice
    # It keeps irrelevent parts in memory but saves time
    product_csr_indices = product_csr_indices[:non_zero_entries_so_far]
    product_csr_data = product_csr_data[:non_zero_entries_so_far]
    return CSR_Mat(
        product_csr_data, product_csr_indices, product_csr_indptrs, M_number_of_rows
    )