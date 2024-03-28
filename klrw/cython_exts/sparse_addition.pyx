import cython
from cython.cimports import cython
import numpy as np
from cython.cimports import numpy as np

from .sparse_csc cimport CSC_Mat

def add(left: CSC_Mat, right: CSC_Mat) -> CSC_Mat:
    assert len(left.indptrs) == len(right.indptrs), "Shapes do not match: different number of columns"
    assert left.number_of_rows == right.number_of_rows, "Shapes do not match: different number of rows"

    sum_csc_indptrs: cython.int[::1] = np.zeros(
        len(left.indptrs), dtype=np.dtype("intc")
    )
    max_non_zero_entries: cython.int = left.nnz() + right.nnz()
    sum_csc_indices: cython.int[::1] = np.zeros(
        max_non_zero_entries, dtype=np.dtype("intc")
    )
    sum_csc_data: object[::1] = np.empty(max_non_zero_entries, dtype=object)

    n: cython.int
    j: cython.int
    indptr1: cython.int
    indptr2: cython.int
    indptr1_end: cython.int
    indptr2_end: cython.int

    non_zero_entries_so_far: cython.int = 0

    for j in range(len(left.indptrs) - 1):
        indptr1 = left.indptrs[j]
        indptr2 = right.indptrs[j]
        indptr1_end = left.indptrs[j + 1]
        indptr2_end = right.indptrs[j + 1]
        while indptr1 != indptr1_end or indptr2 != indptr2_end:
            if indptr1 == indptr1_end:
                sum_csc_data[non_zero_entries_so_far] = right.data[indptr2]
                sum_csc_indices[non_zero_entries_so_far] = right.indices[indptr2]
                non_zero_entries_so_far += 1

                indptr2 += 1

            elif indptr2 == indptr2_end:
                sum_csc_data[non_zero_entries_so_far] = left.data[indptr1]
                sum_csc_indices[non_zero_entries_so_far] = left.indices[indptr1]
                non_zero_entries_so_far += 1

                indptr1 += 1

            elif left.indices[indptr1] == right.indices[indptr2]:
                entry = left.data[indptr1] + right.data[indptr2]
                if not entry.is_zero():
                    sum_csc_data[non_zero_entries_so_far] = entry
                    sum_csc_indices[non_zero_entries_so_far] = left.indices[indptr1]
                    non_zero_entries_so_far += 1
                indptr1 += 1
                indptr2 += 1

            elif left.indices[indptr1] < right.indices[indptr2]:
                sum_csc_data[non_zero_entries_so_far] = left.data[indptr1]
                sum_csc_indices[non_zero_entries_so_far] = left.indices[indptr1]
                non_zero_entries_so_far += 1

                indptr1 += 1

            else:
                sum_csc_data[non_zero_entries_so_far] = right.data[indptr2]
                sum_csc_indices[non_zero_entries_so_far] = right.indices[indptr2]
                non_zero_entries_so_far += 1

                indptr2 += 1

        sum_csc_indptrs[j + 1] = non_zero_entries_so_far

    # Deleting tails of None's in data and zeroes in indices
    sum_csc_data = sum_csc_data[:non_zero_entries_so_far]
    sum_csc_indices = sum_csc_indices[:non_zero_entries_so_far]

    return CSC_Mat(
        sum_csc_data,
        sum_csc_indices,
        sum_csc_indptrs,
        left.number_of_rows,
    )