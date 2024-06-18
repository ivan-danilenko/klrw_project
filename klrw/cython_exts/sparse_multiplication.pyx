# cython: profile=True

import cython
from cython.cimports import cython
import numpy as np
from cython.cimports import numpy as np

from cython.cimports.klrw.cython_exts.sparse_csr import CSR_Mat
from cython.cimports.klrw.cython_exts.sparse_csc import CSC_Mat


def multiplication(left: CSR_Mat, right: CSC_Mat, as_dict: cython.bint = False):
    assert (
        left.number_of_columns == right.number_of_rows
    ), "Shapes do not match: the number of columns in left does not match the number of columns in right."

    if as_dict:
        product_dict = {}
    else:
        # TODO: replace by C++ queues?
        product_csr_data_list = []
        product_csr_indices_list = []
        product_csr_indptrs_list = [0]

    left_number_of_rows: cython.int = len(left.indptrs) - 1
    right_number_of_columns: cython.int = len(right.indptrs) - 1

    i: cython.int
    j: cython.int
    indptr1: cython.int
    indptr2: cython.int
    indptr1_end: cython.int
    indptr2_end: cython.int

    entry_can_be_non_zero: cython.bint = False

    for i in range(left_number_of_rows):
        for j in range(right_number_of_columns):
            indptr1 = left.indptrs[i]
            indptr2 = right.indptrs[j]
            indptr1_end = left.indptrs[i + 1]
            indptr2_end = right.indptrs[j + 1]
            while indptr1 != indptr1_end and indptr2 != indptr2_end:
                if left.indices[indptr1] == right.indices[indptr2]:
                    if not entry_can_be_non_zero:
                        dot_product = left.data[indptr1] * right.data[indptr2]
                        entry_can_be_non_zero = True
                    else:
                        dot_product += left.data[indptr1] * right.data[indptr2]
                    indptr1 += 1
                    indptr2 += 1
                elif left.indices[indptr1] < right.indices[indptr2]:
                    indptr1 += 1
                else:
                    indptr2 += 1

            if entry_can_be_non_zero:
                if not dot_product.is_zero():
                    if as_dict:
                        product_dict[i, j] = dot_product
                    else:
                        product_csr_data_list.append(dot_product)
                        product_csr_indices_list.append(j)
                entry_can_be_non_zero = False
        if not as_dict:
            product_csr_indptrs_list.append(len(product_csr_indices_list))

    if as_dict:
        return product_dict

    product_csr_data = np.array(product_csr_data_list, dtype="O")
    product_csr_indices = np.array(product_csr_indices_list, dtype="intc")
    product_csr_indptrs = np.array(product_csr_indptrs_list, dtype="intc")

    return CSR_Mat(
        product_csr_data,
        product_csr_indices,
        product_csr_indptrs,
        right._number_of_rows(),
    )
