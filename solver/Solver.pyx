import cython

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix, csc_matrix, spmatrix
from scipy.sparse.linalg import cg

from itertools import count
import bisect

from sage.rings.polynomial.polydict import ETuple
from sage.rings.real_mpfr import RealField, RR
from sage.rings.real_double import RDF
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.rings.finite_rings.finite_field_constructor import GF
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.ring import PrincipalIdealDomain
from sage.modules.fg_pid.fgp_module import FGP_Module
from sage.matrix.constructor import matrix
from sage.matrix.matrix_sparse import Matrix_sparse


from KRLWv2 import KRLWAlgebra, KRLWElement, KRLWbraid_one_strand, word_by_extending_permutation


#Making our version of CSR matrices, because scipy rejects working with KRLWElement entries
#Make into structs?
@cython.cclass
class CSR_Mat:
    data : object[::1]
    indices : cython.int[::1]
    indptrs : cython.int[::1]
    number_of_columns : cython.int

    def __init__(self, data, indices, indptrs, number_of_columns):
        assert len(data) == len(indices)
        assert indptrs[0] == 0
        assert indptrs[-1] == len(data)
        #more checks?
        
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
        print(len(self.data),len(self.indices),len(self.indptrs),self.number_of_columns)
    def print_indices(self):
        print(np.asarray(self.indices))
    def print_indptrs(self):
        print(np.asarray(self.indptrs))
        
#    @cython.ccall
#    def to_csc(self):
#        #scipy doesn't support matrix multiplication and conversion to CSR with non-standard coefficients
#        #Nevertheless we can use it to convert from one sparce matrix form to another
#        #Since it's basically resorting indices + some compression.
#        #Scipy will to do it fast
#        M_csc = csc_matrix((range(1,len(self.indices)+1),self.indices,self.indptrs))
#        M_csr = M_csc.tocsr()
#        del M_csc

#        csr_data : object[::1] = np.empty(len(self.data),dtype=object)
#        for i in range(len(self.data)):
#            csr_data[i]= self.data[M_csr.data[i]-1]
#        csr_indices : cython.int[::1] = np.array(M_csr.indices, dtype=np.dtype('intc'))
#        csr_indptrs : cython.int[::1] = np.array(M_csr.indptr, dtype=np.dtype('intc'))

#        return CSR_Mat(csr_data,csr_indices,csr_indptrs)
    
    @cython.ccall
    def is_zero(self, tolerance : cython.double = 0):
        assert tolerance>=0
        i : cython.int
        for i in range(self.nnz()):
            for x,coef in self.data[i].monomial_coefficients().items():
                if tolerance == 0:
                    if not coef.is_zero():
                        return False
                else:
                    scalar : cython.double
                    for __,scalar in coef.iterator_exp_coeff():
                        if scalar > tolerance or scalar<-tolerance:
                            return False
                        
        return True
    
    @cython.ccall
    def is_zero_mod(self, ideal, tolerance : cython.double = 0):
        assert tolerance>=0
        i : cython.int
        for i in range(self.nnz()):
            for x,coef in self.data[i].monomial_coefficients().items():
                if tolerance == 0:
                    if not ideal.reduce(coef).is_zero():
                        return False
                else:
                    scalar : cython.double
                    for __,scalar in ideal.reduce(coef).iterator_exp_coeff():
                        if scalar > tolerance or scalar<-tolerance:
                            return False
                        
        return True

#Making our version of CSC matrices, because scipy rejects working with object entries
@cython.cclass
class CSC_Mat:
    data : object[::1]
    indices : cython.int[::1]
    indptrs : cython.int[::1]
    number_of_rows : cython.int
        
    def __init__(self, data, indices, indptrs, number_of_rows):
        assert len(data) == len(indices)
        assert indptrs[0] == 0
        assert indptrs[-1] == len(data), repr(indptrs[-1]) + " " + repr(len(data))
        #more checks?
        
        self.data = data
        self.indices = indices
        self.indptrs = indptrs
        self.number_of_rows = number_of_rows
        
    def print_sizes(self):
        print(len(self.data),len(self.indices),len(self.indptrs),self.number_of_rows)
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
    @cython.ccall
    def nnz(self) -> cython.int:
        return len(self.indices)
    
    def as_an_array(self):
        N : cython.int = self.number_of_rows
        i : cython.int
        j : cython.int

        A = np.empty((N, len(self.indptrs)-1), dtype=np.dtype('O'))

        for i in range(len(self.indptrs)-1):
            for j in range(self.indptrs[i], self.indptrs[i+1]):
                A[self.indices[j], i] = self.data[j]

        return A

     
    #TODO: return KRLW.zero() if not present
    def __getitem__(self, key):
        """
        Returns (i,j)-th element if present
        Returns None is not present
        """
        assert len(key) == 2
        i : cython.int = key[0]
        j : cython.int = key[1]
        assert i >= 0
        assert i <= self.number_of_rows
        assert j >= 0
        assert j < len(self.indptrs)
        
        ind : cython.int = self.indptrs[j]
        ind_end : cython.int = self.indptrs[j+1]
        while ind != ind_end:
            ii : cython.int = self.indices[ind]
            if ii < i:
                ind += 1
            elif ii == i:
                return self.data[ind]
            else:
                break
        return None
        
    @cython.ccall
    def to_csr(self):# -> CSR_Mat:
        #scipy doesn't support matrix multiplication and conversion to CSR with non-standard coefficients
        #Nevertheless we can use it to convert from one sparce matrix form to another
        #Since it's basically resorting indices + some compression.
        #Scipy will to do it fast
        M_csc = csc_matrix((range(1,len(self.indices)+1),self.indices,self.indptrs),
                           shape = (len(self.indptrs)-1,self.number_of_rows)
                          )
        M_csr = M_csc.tocsr()
        del M_csc

        csr_data : object[::1] = np.empty(len(self.data),dtype=object)
        i : cython.int
        entry : cython.int
        for i in range(self.nnz()):
            entry = M_csr.data[i]
            csr_data[i]= self.data[entry-1]
        csr_indices : cython.int[::1] = M_csr.indices.astype(dtype=np.dtype("intc"))
        csr_indptrs : cython.int[::1] = M_csr.indptr.astype(dtype=np.dtype("intc"))

        return CSR_Mat(csr_data, csr_indices, csr_indptrs, len(self.indptrs)-1)
    
    @cython.ccall
    def is_zero(self, tolerance : cython.double = 0):
        assert tolerance>=0
        i : cython.int
        for i in range(self.nnz()):
            for x,coef in self.data[i].monomial_coefficients().items():
                if tolerance == 0:
                    if not coef.is_zero():
                        return False
                else:
                    scalar : cython.double
                    for __,scalar in coef.iterator_exp_coeff():
                        if scalar > tolerance or scalar<-tolerance:
                            return False
                        
        return True
    
    @cython.ccall
    def is_zero_mod(self, ideal, tolerance : cython.double = 0):
        assert tolerance>=0
        i : cython.int
        for i in range(self.nnz()):
            for x,coef in self.data[i].monomial_coefficients().items():
                if tolerance == 0:
                    if not ideal.reduce(coef).is_zero():
                        return False
                else:
                    scalar : cython.double
                    for __,scalar in ideal.reduce(coef).iterator_exp_coeff():
                        if scalar > tolerance or scalar<-tolerance:
                            return False
                        
        return True
    
    def squares_to_zero(self, mod_ideal = None):
        csr = self.to_csr()
        
        assert self.number_of_rows == len(self.indptrs) - 1
        N : cython.int = self.number_of_rows
        
        i : cython.int
        j : cython.int
        indptr1 : cython.int
        indptr2 : cython.int
        indptr1_end : cython.int
        indptr2_end : cython.int

        non_zero_entries_so_far : cython.int = 0
        entry_can_be_non_zero : cython.bint = False

        for i in range(N):
            for j in range(N):
                indptr1 = csr.indptrs[i]
                indptr2 = self.indptrs[j]
                indptr1_end = csr.indptrs[i+1]
                indptr2_end = self.indptrs[j+1]
                while indptr1 != indptr1_end and indptr2 != indptr2_end:
                    if csr.indices[indptr1] == self.indices[indptr2]:
                        if not entry_can_be_non_zero:
                            dot_product : KRLWElement = csr.data[indptr1] * self.data[indptr2]
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
                    if mod_ideal == None:
                        if not dot_product.is_zero():
                            return False
                    else:
                        for coef in dot_product.monomial_coefficients().values():
                            if mod_ideal.reduce(coef)!=0:
                                return False
                            
        #if never seen a non-zero matrix element                    
        return True

    
    
#might give a time increase in speed, but will not give errors and indexing from the tail
@cython.boundscheck(False)
@cython.wraparound(False)
def multiply(M_csr : CSR_Mat,
             N_csc : CSC_Mat,
            ) -> CSR_Mat:
    
    M_number_of_rows : cython.int = len(M_csr.indptrs) - 1
    N_number_of_columns : cython.int = len(N_csc.indptrs) - 1
    
    product_csr_indptrs : cython.int[::1] = np.zeros(M_number_of_rows+1, dtype=np.dtype('intc'))
    max_non_zero_entries : cython.int = len(M_csr.indices)*len(N_csc.indices)
    product_csr_indices : cython.int[::1] = np.zeros(max_non_zero_entries, dtype=np.dtype('intc'))
    product_csr_data : object[::1] = np.empty(max_non_zero_entries,dtype=object)
        
    i : cython.int
    j : cython.int
    indptr1 : cython.int
    indptr2 : cython.int
    indptr1_end : cython.int
    indptr2_end : cython.int
    
    non_zero_entries_so_far : cython.int = 0
    entry_can_be_non_zero : cython.bint = False
        
    for i in range(M_number_of_rows):
        for j in range(N_number_of_columns):
            indptr1 = M_csr.indptrs[i]
            indptr2 = N_csc.indptrs[j]
            indptr1_end = M_csr.indptrs[i+1]
            indptr2_end = N_csc.indptrs[j+1]
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

        product_csr_indptrs[i+1] = non_zero_entries_so_far
        
    # Deleting tails of None's in data and zeroes in indices
    # product_csr_indices = np.resize(product_csr_indices,(non_zero_entries_so_far,))
    # product_csr_data = np.resize(product_csr_data,(non_zero_entries_so_far,))
    # Deleting tails of None's in data and zeroes in indices
    # Since we don't want to recreate the array, we use a slice
    # It keeps irrelevent parts in memory but saves time
    product_csr_indices = product_csr_indices[:non_zero_entries_so_far]
    product_csr_data = product_csr_data[:non_zero_entries_so_far]
    return CSR_Mat(product_csr_data, product_csr_indices, product_csr_indptrs, M_number_of_rows)


@cython.cfunc
def mod_h_sq(exponents : ETuple, order : cython.int) -> cython.bint:
    return <cython.bint>(exponents[1] == order)

@cython.cfunc
def mod_u_sq_h(exponents : ETuple, order : cython.int) -> cython.bint:
    return <cython.bint>(exponents[0] == order and exponents[1] == 0)

@cython.cclass
class Solver:
    KRLW : KRLWAlgebra
    _tolerance : cython.double
    verbose : cython.bint
    N : typing.Optional[cython.int]
    d0_csc : CSC_Mat
    d1_csc : CSC_Mat
    number_of_variables : typing.Optional[cython.int]
    
    def __init__(self, KRLW, verbose=True):
        self.KRLW = KRLW
        self._tolerance = 0
        self.verbose = verbose
        self.N : cython.int = None
        self.number_of_variables = None
            
    def KRLW_algebra(self):
        """
        Returns the KRLW algebra.
        """
        return self.KRLW
    
    def tolerance(self):
        return self._tolerance

    @cython.ccall
    def d0(self):
        return self.d0_csc
                    
    def set_d0(self, d0_csc : CSC_Mat):
        #remember the number of T-branes if not known before
        if self.N != None:
            assert self.N==d0_csc.number_of_rows, "Number of thimbles must match."
            assert len(d0_csc.indptrs)==d0_csc.number_of_rows+1, "The differential matrix must be square."
        else:
            self.N = d0_csc.number_of_rows
        
        self.d0_csc = d0_csc


    def set_d1(self, d1_csc : CSC_Mat, number_of_variables):
        #remember the number of T-branes if not known before
        if self.N != None:
            assert self.N==d1_csc.number_of_rows, "Number of thimbles must match."
            assert len(d1_csc.indptrs)==d1_csc.number_of_rows+1, "The differential matrix must be square."
        else:
            self.N = d1_csc.number_of_rows
        
        self.d1_csc = d1_csc
        self.number_of_variables = number_of_variables
        
        
    def _d1_(self):
        return self.d1_csc
    
    #@cython.ccall
    #@cython.cfunc
    def make_system_for_corrections(self, multiplier : object, graded_type : str, order : cython.int = 1):
        #d0_csc = self.d0_csc
        #d1_csc = self.d1_csc
        d0_csr : CSR_Mat = self.d0_csc.to_csr()
        d1_csr : CSR_Mat = self.d1_csc.to_csr()
        if self.verbose:
            print("Making the system")

        N : cython.int = self.N

        if graded_type == "h^order":
            condition = mod_h_sq
        elif graded_type == "u^order*h^0":
            condition = mod_u_sq_h
        else:
            raise ValueError("Unknown modulo type")

        A_csr_data_list : list = []
        A_csr_indices_list : list = []
        A_csr_indptrs_list : list = [0]
        b_data_list : list = []
        b_indices_list : list  = []

        n : cython.int
        i : cython.int
        j : cython.int
        indptr1 : cython.int
        indptr2 : cython.int
        indptr1_end : cython.int
        indptr2_end : cython.int
        equations_so_far : cython.int = 0
        entries_so_far : cython.int = 0
        entry_can_be_non_zero : cython.bint = False

        ij_dict : dict
        #row : dict
        index = 0
        for i in range(N):
            for j in range(N):
                #computing d0*d1 part
                #keep (i,j)-th entry as dictionary {(KRLW_symmetric_dots_and_u_hbar,KRLWbraid): dict_representing_row }
                #(KRLW_dots_and_u_hbar,KRLWbraid) parametrizes rows in the system
                #dict_representing_row is a dictionary {variable_number,system_coefficient}
                indptr1 = d0_csr.indptrs[i]
                indptr2 = self.d1_csc.indptrs[j]
                #indptr2 = d1_csc.indptrs[j]
                indptr1_end = d0_csr.indptrs[i+1]
                indptr2_end = self.d1_csc.indptrs[j+1]
                #indptr2_end = d1_csc.indptrs[j+1]
                while indptr1 != indptr1_end and indptr2 != indptr2_end:
                    #if d0_csr.indices[indptr1] == d1_csc.indices[indptr2]:
                    if d0_csr.indices[indptr1] == self.d1_csc.indices[indptr2]:
                        if not entry_can_be_non_zero:
                            ij_dict = {}
                            entry_can_be_non_zero = True

                        d0_element = d0_csr.data[indptr1]
                        #d1_elements = d1_csc.data[indptr2]
                        d1_elements = self.d1_csc.data[indptr2]
                        for n,d1_elem in d1_elements.items():
                            if n == -1:
                                print("+")
                            KRLW_element = multiplier*d0_element*d1_elem
                            for basis_vector,coef in KRLW_element.monomial_coefficients().items():
                                for exp,scalar in coef.iterator_exp_coeff():
                                    if condition(exp, order):
                                        if (exp,basis_vector) in ij_dict:
                                            row = ij_dict[exp,basis_vector]
                                            if n in row:
                                                row[n] += scalar
                                            else:
                                                row[n] = scalar
                                        else:
                                            ij_dict[exp,basis_vector] = {n:scalar}

                        indptr1 += 1
                        indptr2 += 1
                    elif d0_csr.indices[indptr1] < self.d1_csc.indices[indptr2]:
                    #elif d0_csr.indices[indptr1] < d1_csc.indices[indptr2]:
                        indptr1 += 1
                    else:
                        indptr2 += 1

                #computing d1*d0 part
                indptr1 = d1_csr.indptrs[i]
                #indptr2 = d0_csc.indptrs[j]
                indptr2 = self.d0_csc.indptrs[j]
                indptr1_end = d1_csr.indptrs[i+1]
                #indptr2_end = d0_csc.indptrs[j+1]
                indptr2_end = self.d0_csc.indptrs[j+1]
                while indptr1 != indptr1_end and indptr2 != indptr2_end:
                    #if d1_csr.indices[indptr1] == d0_csc.indices[indptr2]:
                    if d1_csr.indices[indptr1] == self.d0_csc.indices[indptr2]:
                        if not entry_can_be_non_zero:
                            ij_dict = {}
                            entry_can_be_non_zero = True

                        d1_elements = d1_csr.data[indptr1]
                        #d0_element = d0_csc.data[indptr2]
                        d0_element = self.d0_csc.data[indptr2]
                        for n in d1_elements:
                            if n == -1:
                                print("+")
                            KRLW_element = multiplier*d1_elements[n]*d0_element
                            for basis_vector,coef in KRLW_element.monomial_coefficients().items():
                                for exp,scalar in coef.iterator_exp_coeff():
                                    if condition(exp, order):
                                        if (exp,basis_vector) in ij_dict:
                                            row = ij_dict[exp,basis_vector]
                                            if n in row:
                                                row[n] += scalar
                                            else:
                                                row[n] = scalar
                                        else:
                                            ij_dict[exp,basis_vector] = {n:scalar}

                        indptr1 += 1
                        indptr2 += 1
                    #elif d1_csr.indices[indptr1] < d0_csc.indices[indptr2]:
                    elif d1_csr.indices[indptr1] < self.d0_csc.indices[indptr2]:
                        indptr1 += 1
                    else:
                        indptr2 += 1

                #if the entry could be non-zero, computing -d0*d0 part
                #don't forget the sign inside!
                #using n=-1 for keeping in a row
                if entry_can_be_non_zero:
                    indptr1 = d0_csr.indptrs[i]
                    #indptr2 = d0_csc.indptrs[j]
                    indptr2 = self.d0_csc.indptrs[j]
                    indptr1_end = d0_csr.indptrs[i+1]
                    #indptr2_end = d0_csc.indptrs[j+1]
                    indptr2_end = self.d0_csc.indptrs[j+1]
                    while indptr1 != indptr1_end and indptr2 != indptr2_end:
                        #if d0_csr.indices[indptr1] == d0_csc.indices[indptr2]:
                        if d0_csr.indices[indptr1] == self.d0_csc.indices[indptr2]:
                            d0_element1 = d0_csr.data[indptr1]
                            #d0_element2 = d0_csc.data[indptr2]
                            d0_element2 = self.d0_csc.data[indptr2]
                            KRLW_element = d0_element1*d0_element2
                            for basis_vector, coef in KRLW_element._monomial_coefficients.items():
                                for exp,scalar in coef.iterator_exp_coeff():
                                    if condition(exp, order):
                                        #here we keep only (exp,basis_vector) in ij_dict, 
                                        #the others should cancel in a consistent system
                                        if (exp,basis_vector) in ij_dict:
                                            row = ij_dict[exp,basis_vector]
                                            if -1 in row:
                                                row[-1] += -scalar
                                            else:
                                                row[-1] = -scalar

                            indptr1 += 1
                            indptr2 += 1
                        #elif d0_csr.indices[indptr1] < d0_csc.indices[indptr2]:
                        elif d0_csr.indices[indptr1] < self.d0_csc.indices[indptr2]:
                            indptr1 += 1
                        else:
                            indptr2 += 1

                if entry_can_be_non_zero:    
                    for __,row in ij_dict.items():
                        variables : list  = list(row.keys())
                        variables.sort()
                        variable_number : cython.int
                        for variable_number in variables:
                            if variable_number == -1:
                                b_data_list.append(row[-1])
                                b_indices_list.append(equations_so_far)
                            else:
                                scalar : cython.int = row[variable_number]
                                A_csr_data_list.append(scalar)
                                A_csr_indices_list.append(variable_number)
                                entries_so_far += 1

                        A_csr_indptrs_list.append(entries_so_far)
                        equations_so_far +=1

                    entry_can_be_non_zero = False

        number_of_equations : cython.int = equations_so_far

        if self.verbose:
            print("We have",number_of_equations,"rows",self.number_of_variables,"columns")

        A_csr = csr_matrix((A_csr_data_list, A_csr_indices_list, A_csr_indptrs_list),
                       shape = (number_of_equations, self.number_of_variables))


        b = csc_matrix((b_data_list, b_indices_list, (0,len(b_indices_list))),
                       shape = (number_of_equations,1))

        if self.verbose:
            print("Number of terms to correct:",len(b.data))

        return A_csr, b

    @cython.cfunc
    def solve_system_for_differential(self, M : spmatrix, bb : spmatrix):
        """
        Returns a tuple (solution : double[::1], is_integral : bool)
        """
        if self.verbose:
            print("Solving the system")

        #scipy conjugate gradients only take dense vectors, so we convert
        #A1 flattens array
        y = bb.todense().A1

        x, exit_code = cg(M,y)
        if self.verbose:
            print("Exit_Code:",exit_code)

        #trying an integer approximation if it works
        x_int = np.rint(x)#.astype(dtype=np.dtype("intc"))
        
        assert np.allclose(M@x_int, y), "Solution is not integral"
        return x_int    

    @cython.cfunc
    def update_differential(self, x : cython.double[::1], multiplier : KRLWElement = 1):
        if self.verbose:
            print("Correcting the differential")

        number_of_columns : cython.int = self.N

        correted_csc_indptrs : cython.int[::1] = np.zeros(number_of_columns+1, dtype=np.dtype('intc'))
        max_non_zero_entries : cython.int = self.d0_csc.nnz() + self.d1_csc.nnz()
        correted_csc_indices : cython.int[::1] = np.zeros(max_non_zero_entries, dtype=np.dtype('intc'))
        correted_csc_data : object[::1] = np.empty(max_non_zero_entries,dtype=object)    

        n : cython.int
        j : cython.int
        indptr1 : cython.int
        indptr2 : cython.int
        indptr1_end : cython.int
        indptr2_end : cython.int

        non_zero_entries_so_far : cython.int = 0

        for j in range(number_of_columns):
            indptr1 = self.d0_csc.indptrs[j]
            indptr2 = self.d1_csc.indptrs[j]
            indptr1_end = self.d0_csc.indptrs[j+1]
            indptr2_end = self.d1_csc.indptrs[j+1]
            while indptr1 != indptr1_end or indptr2 != indptr2_end:
                #TODO: better cases
                if indptr1 == indptr1_end:
                    entry = self.KRLW.zero()
                    for n,d1_entry in self.d1_csc.data[indptr2].items():
                        entry += multiplier*self.KRLW.center(x[n])*d1_entry
                    if not entry.is_zero():
                        correted_csc_data[non_zero_entries_so_far] = entry
                        correted_csc_indices[non_zero_entries_so_far] = self.d1_csc.indices[indptr2]
                        non_zero_entries_so_far += 1

                    indptr2 += 1

                elif indptr2 == indptr2_end:
                    correted_csc_data[non_zero_entries_so_far] = self.d0_csc.data[indptr1]
                    correted_csc_indices[non_zero_entries_so_far] = self.d0_csc.indices[indptr1]
                    non_zero_entries_so_far += 1

                    indptr1 += 1

                elif self.d0_csc.indices[indptr1] == self.d1_csc.indices[indptr2]:
                    entry = self.d0_csc.data[indptr1] 
                    for n,d1_entry in self.d1_csc.data[indptr2].items():
                        entry += multiplier*self.KRLW.center(x[n])*d1_entry
                    if not entry.is_zero():
                        correted_csc_data[non_zero_entries_so_far] = entry
                        correted_csc_indices[non_zero_entries_so_far] = self.d0_csc.indices[indptr1]
                        non_zero_entries_so_far += 1

                    indptr1 += 1
                    indptr2 += 1

                elif self.d0_csc.indices[indptr1] < self.d1_csc.indices[indptr2]:
                    correted_csc_data[non_zero_entries_so_far] = self.d0_csc.data[indptr1]
                    correted_csc_indices[non_zero_entries_so_far] = self.d0_csc.indices[indptr1]
                    non_zero_entries_so_far += 1

                    indptr1 += 1

                else:
                    entry = self.KRLW.zero()
                    for n,d1_entry in self.d1_csc.data[indptr2].items():
                        entry += multiplier*self.KRLW.center(x[n])*d1_entry
                    if not entry.is_zero():
                        correted_csc_data[non_zero_entries_so_far] = entry
                        correted_csc_indices[non_zero_entries_so_far] = self.d1_csc.indices[indptr2]
                        non_zero_entries_so_far += 1

                    indptr2 += 1

            correted_csc_indptrs[j+1] = non_zero_entries_so_far

        #Deleting tails of None's in data and zeroes in indices
        correted_csc_indices = np.resize(correted_csc_indices,(non_zero_entries_so_far,))
        correted_csc_data = np.resize(correted_csc_data,(non_zero_entries_so_far,))

        self.d0_csc = CSC_Mat(correted_csc_data, correted_csc_indices, correted_csc_indptrs, number_of_columns)


    def make_corrections(self, multiplier =1, order: cython.int = 1, graded_type = "h^order"):
        """
        Solves the system (d0 + multiplier*d1)^2 = 0 in a graded component denoted by graded_type and order.
        If graded_type = "h^order" the graded component is all terms with h^order [all powers of u possible]
        If graded_type = "u^order*h^0" the graded component is all terms with u^order*h^0
        """
        A_csr,b = self.make_system_for_corrections(multiplier = multiplier, order = order, graded_type = graded_type)
        
        A_csc = A_csr.tocsc()
        columns_to_remove = 0
        for i in range(len(A_csc.indptr)-1):
            if A_csc.indptr[i] == A_csc.indptr[i+1]:
                columns_to_remove += 1
                print("Zero column in the original matrix:", i)

        if columns_to_remove > 0:
            number_of_old_indices : cython.int = A_csc.shape[1]
            number_of_new_indices : cython.int = A_csc.shape[1]-columns_to_remove
            new_to_old_index = np.zeros(number_of_new_indices, dtype='intc')
            A_csc_indptrs_new = np.zeros(number_of_new_indices + 1, dtype='intc')
            new_indices_so_far : cython.int = 0
            i : cython.int
            for i in range(A_csc.shape[1]):
                # if we don't delete this column
                if A_csc.indptr[i] < A_csc.indptr[i+1]:
                    new_to_old_index[new_indices_so_far] = i
                    A_csc_indptrs_new[new_indices_so_far + 1] = A_csc.indptr[i+1]
                    new_indices_so_far += 1
            A = csc_matrix(
                (A_csc.data, A_csc.indices, A_csc_indptrs_new),
                shape = (A_csc.shape[0], number_of_new_indices)
            )
        else:
            A = A_csc

        from pickle import dump
        with open("matrix", "wb") as f:
            dump(A, file=f)
        with open("vector", "wb") as f:
            dump(b, file=f)

        if self.verbose:
            print("Transfoming to a symmetric square system")
        A_tr = A.transpose()
        M = A_tr @ A
        bb = A_tr @ b

        for i in range(len(M.indptr)-1):
            if M.indptr[i] == M.indptr[i+1]:
                print("Zero column in the square matrix:", i)
        
        x = self.solve_system_for_differential(M,bb)
        del M,bb

        if columns_to_remove > 0:
            x_modified = np.zeros(number_of_old_indices, dtype='d')
            i : cython.int
            for i in range(number_of_new_indices):
                x_modified[new_to_old_index[i]] = x[i]
            x = x_modified
        
        #if still working over Z, comparison is exact
        if self.KRLW.center.base_ring() == ZZ:
            assert np.array_equal(A_csc@x.astype(np.dtype("intc")), b.todense().A1), "Not a solution!"
        else:
            assert np.allclose(A_csc@x, b.todense().A1, atol = self.tolerance()), "Not a solution!"
        
        if self.verbose:
            print("Found a solution!")
            nnz = sum(1 for a in x.flat if a != 0)
            print("Correcting {} matrix elements".format(nnz))
        del A,b
        
        #self.KRLW_algebra().center(multiplier) in case the ground ring changed
        multiplier=self.KRLW_algebra().center(multiplier)
        self.update_differential(x, multiplier)
      
    def check_d0(self):
        d0_squared_csr = multiply(self.d0_csc.to_csr(),self.d0_csc)
        print("d0 squares to zero:", d0_squared_csr.is_zero(self.tolerance()))

        hucenter = self.KRLW.center.ideal([self.KRLW.u,self.KRLW.hbar])
        print("d0 squares to zero mod (u,h):", d0_squared_csr.is_zero_mod(hucenter, self.tolerance()))
        
        husqcenter = self.KRLW.center.ideal([self.KRLW.u**2,self.KRLW.hbar])
        print("d0 squares to zero mod (u^2,h):", d0_squared_csr.is_zero_mod(husqcenter, self.tolerance()))
        
        hucucenter = self.KRLW.center.ideal([self.KRLW.u**3,self.KRLW.hbar])
        print("d0 squares to zero mod (u^3,h):", d0_squared_csr.is_zero_mod(hucucenter, self.tolerance()))
        
        hcenter = self.KRLW.center.ideal([self.KRLW.hbar])
        print("d0 squares to zero mod h:", d0_squared_csr.is_zero_mod(hcenter, self.tolerance()))

        hsqcenter = self.KRLW.center.ideal([self.KRLW.hbar**2])
        print("d0 squares to zero mod h^2:", d0_squared_csr.is_zero_mod(hsqcenter, self.tolerance()))
        
    def d0_squares_to_zero(self):
        d0_squared_csr = multiply(self.d0_csc.to_csr(),self.d0_csc)
        d0_squared_is_zero = d0_squared_csr.is_zero(self.tolerance())
        print("d0 squares to zero:", d0_squared_is_zero)
        return d0_squared_is_zero

    def d0_squares_to_zero_mod(self, ideal):
        d0_squared_csr = multiply(self.d0_csc.to_csr(),self.d0_csc)
        d0_squared_is_zero = d0_squared_csr.is_zero_mod(ideal, tolerance=self.tolerance())
        print("d0 squares to zero modulo the ideal:", d0_squared_is_zero)
        return d0_squared_is_zero

        
        
class TThimble:
    __slots__ = ('segment', 'hom_deg', 'equ_deg', 'order')
    
    def __init__(self, segment, hom_deg, equ_deg, order):
        self.segment = segment
        self.hom_deg = hom_deg
        self.equ_deg = equ_deg
        self.order = order
        
    def __repr__(self):
        return "T-Thimble in " + self.segment.__repr__() + " segment with "+ self.hom_deg.__repr__() + " cohomological degree and " + self.equ_deg.__repr__() + " equivariant degree, on " + self.order.__repr__() + " position"
    
    
class ProductThimbles:
    '''
    colored_state is a list or tuple where i-th element is the position of the i-th moving strand
    frozenset(...) of this list gives the (uncolored) state in KRLW elements
    order is a tuple of how many intersection points are to the left for each intersection
    intersection_indices is a tuple to remember the indices of thimbles and access information about them
    '''
    __slots__ = ('colored_state', 
                 'hom_deg', 
                 'equ_deg', 
                 'order', 
                 'next_thimble_strand_number',
                 'next_thimble_position', 
                 'intersection_indices')
    
    
    def __init__(self, 
                 colored_state : list | tuple, 
                 hom_deg, equ_deg, 
                 order : list | tuple, 
                 next_thimble_strand_number, 
                 next_thimble_position, 
                 intersection_indices):
        self.colored_state = colored_state
        self.hom_deg = hom_deg
        self.equ_deg = equ_deg
        self.order = order
        self.next_thimble_strand_number = next_thimble_strand_number
        self.next_thimble_position = next_thimble_position
        self.intersection_indices = intersection_indices
        
    def __repr__(self):
        return "Product of T-Thimbles: " + self.colored_state.__repr__() + ", " + self.hom_deg.__repr__() + ", " + self.equ_deg.__repr__() + ", " + self.order.__repr__() + ", " + self.equ_deg.__repr__() + ", " + self.next_thimble_strand_number.__repr__() + ", " + self.next_thimble_position.__repr__() + ", " + self.intersection_indices.__repr__() 

    def uncolored_state(self):
        return frozenset(self.colored_state)


class InitialData:
    def __init__(self, number_of_punctures, number_of_E_branes):
        self.n = number_of_punctures
        self.k = number_of_E_branes
        
        E_brane_cyclic = [0,1,2,3]
        E_brane_intersections = [[0],
                                 [1,3],
                                 [2]
                                ]
        #Warning about equivariant grading: it depends on the convention if we read the braid top-to-bottom or bottom-to-top
        #These gradings work if we take the grading conventions from https://arxiv.org/pdf/2305.13480.pdf
        #but read braids from bottom to top
        E_branes_intersections_data = {0 : TThimble(segment = 0, hom_deg = 0, equ_deg = 0, order = None), 
                                       1 : TThimble(segment = 1, hom_deg = -1, equ_deg = 1, order = None),
                                       2 : TThimble(segment = 2, hom_deg = 0, equ_deg = 1, order = None),
                                       3 : TThimble(segment = 1, hom_deg = 1, equ_deg = 0, order = None)
                                      }
        E_brane_length = len(E_brane_intersections)
        intersection_points_in_E_brane = sum(len(l) for l in E_brane_intersections)
        #Here we keep the intesection points separated into bins between the punctures
        self.intersections = [list() for i in range(number_of_punctures+1)] 
        self.intersections_data = {}

        assert number_of_E_branes*(E_brane_length-1)<=number_of_punctures
        
        self.branes = [[j+i*len(E_brane_cyclic) for j in E_brane_cyclic] for i in range(number_of_E_branes)]
        
        for copy_number in range(number_of_E_branes):
            for j in range(E_brane_length):
                for i in E_brane_intersections[j]:
                    new_point = i + copy_number*intersection_points_in_E_brane
                    segment = j + copy_number*(E_brane_length-1)
                    hom_degree = E_branes_intersections_data[i].hom_deg
                    equ_degree = E_branes_intersections_data[i].equ_deg
                    self.intersections[segment].append(new_point)
                    self.intersections_data[new_point] = TThimble(segment = segment, hom_deg = hom_degree, 
                                                                  equ_deg = equ_degree, order = None)

        #keeps track of branch in log(yi-yj)
        #initially everything in the principal branch
        self.pairwise_equ_deg = {}
        for b1 in range(len(self.branes)):
            for b0 in range(b1):
                for pt0 in self.branes[b0]:
                    for pt1 in self.branes[b1]:
                        self.pairwise_equ_deg[pt0,pt1] = 0
                    
        self.number_of_intersections = number_of_E_branes*intersection_points_in_E_brane

        self.differential = None
        self.thimbles = None
        
    
    def apply_s(self, i):
        assert i!=0
        if i>0:
            sign = +1
        else:
            sign = -1
            i = -i
        assert i<len(self.intersections)
        
        new_points_left = []
        new_points_right = []
        for point in self.intersections[i]:
            #we add two new points, one in the segment on the left, the other in the segment on the right
            left_point = self.number_of_intersections
            right_point = self.number_of_intersections + 1
            new_points_left.append(left_point)
            new_points_right.append(right_point)
            self.number_of_intersections += 2
            
            hom_deg = self.intersections_data[point].hom_deg
            equ_deg = self.intersections_data[point].equ_deg
            if sign == +1:
                self.intersections_data[left_point] = TThimble(segment = i-1, hom_deg = hom_deg,
                                                               equ_deg = equ_deg-1, order = None)
                self.intersections_data[right_point] = TThimble(segment = i+1, hom_deg = hom_deg,
                                                                equ_deg = equ_deg, order = None)
                self.intersections_data[point] = TThimble(segment = i, hom_deg = hom_deg+1, 
                                                          equ_deg = equ_deg-1, order = None)
            else:
                self.intersections_data[left_point] = TThimble(segment = i-1, hom_deg = hom_deg,
                                                               equ_deg = equ_deg, order = None)
                self.intersections_data[right_point] = TThimble(segment = i+1, hom_deg = hom_deg,
                                                                equ_deg = equ_deg+1, order = None)
                self.intersections_data[point] = TThimble(segment = i, hom_deg = hom_deg-1, 
                                                          equ_deg = equ_deg+1, order = None)
            
            #now modify the parts on the branes
            #using "b in range(...)" instead of "b in self.branes" because we need to modify elements of self.branes 
            for b in range(len(self.branes)):
                if point in self.branes[b]:
                    ind = self.branes[b].index(point)
                    #if odd, so goes "down" before cohomological shift
                    if ind%2:
                        if sign == +1:
                            self.branes[b] = self.branes[b][:ind] + [left_point,point,right_point] + self.branes[b][ind+1:]
                        else:
                            self.branes[b] = self.branes[b][:ind] + [right_point,point,left_point] + self.branes[b][ind+1:]
                    #if even, so goes "up" before cohomological shift
                    else:
                        if sign == +1:
                            self.branes[b] = self.branes[b][:ind] + [right_point,point,left_point] + self.branes[b][ind+1:]
                        else:
                            self.branes[b] = self.branes[b][:ind] + [left_point,point,right_point] + self.branes[b][ind+1:]

        #now modify the pairwise contributions to the equivariant degree
        #we do it only on this step because of the case when both points are on i-th segement
        old_keys = [x for x in self.pairwise_equ_deg]
        for pts in old_keys:
            #pts is a 2-tuple
            if_pt_on_the_segment = [p in self.intersections[i] for p in pts]
            #if both points are on the segment
            if if_pt_on_the_segment[0] and if_pt_on_the_segment[1]:
                #find the places of pt0,pt1 and corresponding left&right points
                js = [self.intersections[i].index(p) for p in pts]

                left_points = [new_points_left[j] for j in js]
                right_points = [new_points_right[j] for j in js]

                self.pairwise_equ_deg[left_points[0],left_points[1]] = self.pairwise_equ_deg[pts]
                self.pairwise_equ_deg[right_points[0],right_points[1]] = self.pairwise_equ_deg[pts]
                #for many cases the value depents on relative positions of points on the segment
                if js[0]<js[1]:
                    self.pairwise_equ_deg[left_points[0],pts[1]] = self.pairwise_equ_deg[pts]
                    self.pairwise_equ_deg[left_points[0],right_points[1]] = self.pairwise_equ_deg[pts]
                    self.pairwise_equ_deg[pts[0],left_points[1]] = self.pairwise_equ_deg[pts] + sign
                    self.pairwise_equ_deg[pts[0],right_points[1]] = self.pairwise_equ_deg[pts]
                    self.pairwise_equ_deg[right_points[0],left_points[1]] = self.pairwise_equ_deg[pts] + sign
                    self.pairwise_equ_deg[right_points[0],pts[1]] = self.pairwise_equ_deg[pts] + sign
                else:
                    self.pairwise_equ_deg[left_points[0],pts[1]] = self.pairwise_equ_deg[pts] + sign
                    self.pairwise_equ_deg[left_points[0],right_points[1]] = self.pairwise_equ_deg[pts] + sign
                    self.pairwise_equ_deg[pts[0],left_points[1]] = self.pairwise_equ_deg[pts] 
                    self.pairwise_equ_deg[pts[0],right_points[1]] = self.pairwise_equ_deg[pts] + sign
                    self.pairwise_equ_deg[right_points[0],left_points[1]] = self.pairwise_equ_deg[pts]
                    self.pairwise_equ_deg[right_points[0],pts[1]] = self.pairwise_equ_deg[pts]
                #we change this the last because it's used in all the other cases
                self.pairwise_equ_deg[pts] += sign
            
            elif if_pt_on_the_segment[0]: #and p1 is not on the segment from if statement above
                #find the place of pt0 and corresponding left&right points
                j = self.intersections[i].index(pts[0])
                left_point = new_points_left[j]
                right_point = new_points_right[j]
                #keep self.pairwise_equ_deg[pt0,pt1] the same
                self.pairwise_equ_deg[left_point,pts[1]] = self.pairwise_equ_deg[pts]
                self.pairwise_equ_deg[right_point,pts[1]] = self.pairwise_equ_deg[pts]
                    
            elif if_pt_on_the_segment[1]: #and p1 is not on the segment from if statement above
                #find the place of pt0 and corresponding left&right points
                j = self.intersections[i].index(pts[1])
                left_point = new_points_left[j]
                right_point = new_points_right[j]
                #keep self.pairwise_equ_deg[pt0,pt1] the same
                self.pairwise_equ_deg[pts[0],left_point] = self.pairwise_equ_deg[pts]
                self.pairwise_equ_deg[pts[0],right_point] = self.pairwise_equ_deg[pts]


        self.intersections[i-1] = self.intersections[i-1] + new_points_left
        self.intersections[i+1] = new_points_right + self.intersections[i+1]
        self.intersections[i] = self.intersections[i][::-1]
        
        self.simplify_branes()
        
    def simplify_branes(self):
        #using "b in range(...)" instead of "b in self.branes" because we need to modify elements of self.branes 
        for b in range(len(self.branes)):
            current_index = 0
            #use this instead of for because we might need to decrease current_index in the process
            #and change the length of self.branes[b]
            while current_index != len(self.branes[b]):
                #if not the end
                if current_index != len(self.branes[b]) - 1:
                    next_index = current_index + 1
                    current_point = self.branes[b][current_index]
                    next_point = self.branes[b][next_index]
                    
                    if self.intersections_data[current_point].segment == self.intersections_data[next_point].segment:
                        self.branes[b] = self.branes[b][:current_index] + self.branes[b][next_index+1:]
                        #if cancellation happened, there is a chance the provious point can be cancelled now 
                        current_index -= 1
                    else:
                        current_index += 1
                
                #take cyclicity into account
                else:
                    current_point = self.branes[b][current_index]
                    next_point = self.branes[b][0]
                    if self.intersections_data[current_point].segment == self.intersections_data[next_point].segment:
                        #delete the first and the last element
                        #we also need to keep even positions even and odd ones odd
                        #to make sure going upwards and downwards are consistent
                        #so we move self.branes[b][1] to the end
                        self.branes[b] = self.branes[b][2:current_index] + [self.branes[b][1]]
                        #if cancellation happened, there is a chance the provious point can be cancelled now 
                        #also the numeration shifts by 1, because we removed two elements in the beginning
                        #[one completely removed, another moved to the end]
                        current_index -= 3
                    else:
                        current_index += 1
        
    def apply_braid(self, braid : list):
        for b in braid:
            self.apply_s(b)
            
#    def tensor_product_index(self, indices : list):
#        """
#        Indices in a lexmin order on the tensor product basis
#        """
#        assert len(indices) == len(self.branes)
#        for ind,br in zip(indices,self.branes):
#            assert ind < len(br)
#        
#        tensor_index = 0
#        for ind, br in zip(indices[-1::-1],self.branes[-1::-1]):
#            tensor_index *= len(br)
#            tensor_index += ind
#            
#        return tensor_index

    def one_dimentional_differential_initial(self, i):
        KRLW_one_strand = KRLWAlgebra(ZZ,u_name="u",hbar_name="h", n=self.n, k=1, skip_computing_basis=True, warnings=True)
        brane = self.branes[i]

        entries_so_far = 0
        column_number = 0
        d0_dict = {}
        for current_index in range(len(brane)):
            if current_index != len(brane)-1:
                next_index = current_index + 1
                have_endpoint = False
            #if we are at the end of line, we return to the beginning
            else:
                next_index = 0
                have_endpoint = True

            current_point = brane[current_index]
            next_point = brane[next_index]

            next_hom_deg = self.intersections_data[next_point].hom_deg
            next_segment = self.intersections_data[next_point].segment
            current_hom_deg = self.intersections_data[current_point].hom_deg
            current_segment = self.intersections_data[current_point].segment
            #shift by 1 in segments because numbering of strands in KRLWAlgebra starts with 1.

            if next_hom_deg == current_hom_deg - 1:
                KRLWbraid = KRLWbraid_one_strand(top_state=current_segment+1,
                                                 bottom_state=next_segment+1, 
                                                 xpart=None, N=self.n+1, check=False
                                                )
                
                next_equ_deg = self.intersections_data[next_point].equ_deg
                current_equ_deg = self.intersections_data[current_point].equ_deg
                #print(current_point, next_point, KRLWbraid, KRLWbraid.degree())
                assert next_equ_deg - current_equ_deg == KRLWbraid.degree(), current_equ_deg.__repr__() + " " + next_equ_deg.__repr__() + " " +  KRLWbraid.degree().__repr__()
                
                KRLWElement = KRLW_one_strand.monomial(KRLWbraid)
                if have_endpoint:
                    #we have sign (-1)^{(n/2)+1}
                    if len(brane) % 4 == 0:
                        #???modify the sign?
                        KRLWElement *= ZZ(-1)
                d0_dict[next_index,current_index] = KRLWElement
            elif next_hom_deg == current_hom_deg + 1:
                KRLWbraid = KRLWbraid_one_strand(top_state=next_segment+1,
                                                 bottom_state=current_segment+1, 
                                                 xpart=None, N=self.n+1, check=False
                                                )
                
                next_equ_deg = self.intersections_data[next_point].equ_deg
                current_equ_deg = self.intersections_data[current_point].equ_deg
                #print(current_point, next_point, KRLWbraid, KRLWbraid.degree())
                assert current_equ_deg - next_equ_deg == KRLWbraid.degree(), current_equ_deg.__repr__() + " " + next_equ_deg.__repr__() + " " +  KRLWbraid.degree().__repr__()
                
                KRLWElement = KRLW_one_strand.monomial(KRLWbraid)
                if have_endpoint:
                    #we have sign (-1)^{(n/2)+1}
                    if len(brane) % 4 == 0:
                        #???modify the sign?
                        KRLWElement *= ZZ(-1)
                d0_dict[current_index,next_index] = KRLWElement
            else:
                raise ValueError("Cohomological degrees differ by an unexpected value")
                
        number_of_columns = len(brane)
        #on average, exactly one term in a column/row
        number_of_entries = number_of_columns
        d0_csc_data = np.empty(number_of_entries, dtype = "O")
        d0_csc_indices = np.zeros(number_of_entries, dtype = "intc")
        d0_csc_indptrs = np.zeros(number_of_columns+1, dtype = "intc")

        entries_so_far = 0
        current_j = 0
        for i,j in sorted(d0_dict.keys(), key=lambda x: (x[1],x[0])):
            for k in range(current_j+1,j+1):
                d0_csc_indptrs[k] = entries_so_far
            current_j = j
            d0_csc_data[entries_so_far] = d0_dict[i,j]
            d0_csc_indices[entries_so_far] = i
            entries_so_far += 1
        for k in range(current_j+1,number_of_columns+1):
            d0_csc_indptrs[k] = entries_so_far

        d0_csc = CSC_Mat(data = d0_csc_data, indices = d0_csc_indices, indptrs = d0_csc_indptrs, number_of_rows = number_of_columns)

        return d0_csc
    
    def one_dimentional_differential_corrections(self, i, order):
        KRLW_one_strand = KRLWAlgebra(ZZ,u_name="u",hbar_name="h", n=self.n, k=1, skip_computing_basis=True, warnings=True)
        brane = self.branes[i]
        
        #we organize points/T-thimbles in the brane by their cohomological degree as a dictionary {hom_deg:(index_in_brane,TThimble)}
        points_by_hom_degree = {}
        for index in range(len(brane)):
            point = brane[index]
            thimble = self.intersections_data[point]
            hom_deg = thimble.hom_deg
            
            if hom_deg in points_by_hom_degree:
                points_by_hom_degree[hom_deg].append((index, thimble))
            else:
                points_by_hom_degree[hom_deg] = [(index, thimble)]
                
        d1_dict = {}
        variable_index = 0
        for hom_deg in sorted(points_by_hom_degree.keys()):
            #terms of differential exist only between adjacent coh degrees
            if hom_deg+1 in points_by_hom_degree:
                thimbles_of_hom_deg = points_by_hom_degree[hom_deg]
                thimbles_of_hom_deg_plus_one = points_by_hom_degree[hom_deg+1]
                
                for index0,thimble0 in thimbles_of_hom_deg:
                    for index1,thimble1 in thimbles_of_hom_deg_plus_one:
                        top_state = frozenset({thimble1.segment+1})
                        bot_state = frozenset({thimble0.segment+1})
                        equivariant_degree = thimble0.equ_deg - thimble1.equ_deg
                        #equivariant_degree = thimble1.equ_deg - thimble0.equ_deg 
                        entry = KRLW_one_strand.basis_in_graded_component(top_state=top_state, bot_state=bot_state,
                                                                          equivariant_degree=equivariant_degree,
                                                                          number_of_dots=order, reverse=False)
                        assert len(entry)<=1, "Too many possible corrections for one strand case."
                        if entry:
                            #this is the only element
                            elem = entry[0]
                            d1_dict[index0,index1] = {variable_index: elem}
                            variable_index += 1
        
        #TODO: make a function "from dict" for csc matrices
        number_of_columns = len(brane)
        number_of_entries = len(d1_dict)
        d1_csc_data = np.empty(number_of_entries, dtype = "O")
        d1_csc_indices = np.zeros(number_of_entries, dtype = "intc")
        d1_csc_indptrs = np.zeros(number_of_columns+1, dtype = "intc")
        
        entries_so_far = 0
        current_j = 0
        for i,j in sorted(d1_dict.keys(), key=lambda x: (x[1],x[0])):
            for k in range(current_j+1,j+1):
                d1_csc_indptrs[k] = entries_so_far
            current_j = j
            #entries_so_far becomes the index of a new defomation variable
            d1_csc_data[entries_so_far] = d1_dict[i,j]
            d1_csc_indices[entries_so_far] = i
            entries_so_far += 1
        for k in range(current_j+1,number_of_columns+1):
            d1_csc_indptrs[k] = entries_so_far
            
        d1_csc = CSC_Mat(data = d1_csc_data, indices = d1_csc_indices, indptrs = d1_csc_indptrs, number_of_rows = number_of_columns)
        
        return d1_csc

    def differential_u_corrections(self, thimbles, order, k, d_csc):
        KRLW = KRLWAlgebra(ZZ,u_name="u",hbar_name="h", n=self.n, k=k, skip_computing_basis=True, warnings=True)
        
        #we organize points/T-thimbles in the brane by their cohomological degree as a dictionary {hom_deg:(index_in_brane,TThimble)}
        points_by_hom_degree = {}
        for index,thimble in thimbles.items():
            hom_deg = thimble.hom_deg
            
            if hom_deg in points_by_hom_degree:
                points_by_hom_degree[hom_deg].append((index, thimble))
            else:
                points_by_hom_degree[hom_deg] = [(index, thimble)]
                
        d1_dict = {}
        variable_index = 0
        for hom_deg in sorted(points_by_hom_degree.keys()):
            #terms of differential exist only between adjacent coh degrees
            if hom_deg+1 in points_by_hom_degree:
                thimbles_of_hom_deg = points_by_hom_degree[hom_deg]
                thimbles_of_hom_deg_plus_one = points_by_hom_degree[hom_deg+1]
                
                for index0,thimble0 in thimbles_of_hom_deg:
                    for index1,thimble1 in thimbles_of_hom_deg_plus_one:
                        top_state = frozenset(thimble1.colored_state)
                        bot_state = frozenset(thimble0.colored_state)
                        equivariant_degree = thimble0.equ_deg - thimble1.equ_deg
                        #equivariant_degree = thimble1.equ_deg - thimble0.equ_deg
                        entry = KRLW.basis_in_graded_component(top_state=top_state, bot_state=bot_state,
                                                               equivariant_degree=equivariant_degree,
                                                               number_of_dots=order)

                        existing_entry = d_csc[index0,index1]
                        if existing_entry is not None:
                            if not isinstance(existing_entry, int):
                                for braid, poly in existing_entry:
                                    for exp, _ in poly.iterator_exp_coeff():
                                        dots_exp = ETuple((0,0)) + exp[2:]
                                        existing_term = KRLW.term(index=braid, coeff=KRLW.center.monomial(*dots_exp))
                                        try:
                                            entry.remove(existing_term)
                                        except:
                                            pass # we ignore if the term is not found

                        if entry:
                            #only if we have new elements
                            d1_dict[index0,index1] = {}
                            for elem in entry:
                                d1_dict[index0,index1][variable_index] = elem
                                variable_index += 1
        
        #TODO: make a function "from dict" for csc matrices
        number_of_columns = len(thimbles)
        number_of_entries = len(d1_dict)
        d1_csc_data = np.empty(number_of_entries, dtype = "O")
        d1_csc_indices = np.zeros(number_of_entries, dtype = "intc")
        d1_csc_indptrs = np.zeros(number_of_columns+1, dtype = "intc")
        
        entries_so_far = 0
        current_j = 0
        for i,j in sorted(d1_dict.keys(), key=lambda x: (x[1],x[0])):
            for k in range(current_j+1,j+1):
                d1_csc_indptrs[k] = entries_so_far
            current_j = j
            #entries_so_far becomes the index of a new defomation variable
            d1_csc_data[entries_so_far] = d1_dict[i,j]
            d1_csc_indices[entries_so_far] = i
            entries_so_far += 1
        for k in range(current_j+1,number_of_columns+1):
            d1_csc_indptrs[k] = entries_so_far
            
        d1_csc = CSC_Mat(data = d1_csc_data, indices = d1_csc_indices, indptrs = d1_csc_indptrs, number_of_rows = number_of_columns)

        #print(d1_csc_data)
        
        return d1_csc, variable_index
    
    def one_dimentional_differential(self, i):
        print("----------Making the initial approximation----------")
        d0_csc = self.one_dimentional_differential_initial(i)
        KRLW = KRLWAlgebra(ZZ,u_name="u",hbar_name="h", n=self.n, k=1, skip_computing_basis=True, warnings=True)
        S = Solver(KRLW)
        S.set_d0(d0_csc)
        
        for order in count(start = 1):
            if S.d0_squares_to_zero():
                break
            print("----------Correcting order {} in u for brane {}----------".format(order,i))
            d1_csc = self.one_dimentional_differential_corrections(i, order=order)
            S.set_d1(d1_csc, number_of_variables = len(d1_csc._data()))
            multiplier = (S.KRLW_algebra().u)**order
            S.make_corrections(multiplier = multiplier, order = order, graded_type = "u^order*h^0")
                
        return S.d0_csc

    def make_differential(self):
        assert self.branes, "There has to be at least one E-brane."
        
        #assign all intersection points order, i.e. the number of other intersection points to the left
        ind = 0
        for seg in self.intersections:
            for pt in seg:
                self.intersections_data[pt].order = ind
                ind = ind + 1
                
        #the differential for the zeroth brane
        d_csc_current = self.one_dimentional_differential(0)

        #print(np.asarray(d_csc_current._data()))
        #print(np.asarray(d_csc_current._indices()))
        #print(np.asarray(d_csc_current._indptrs()))
        #print(d_csc_current._number_of_rows())
        
        thimbles = {}
        for index,pt in zip(count(),self.branes[0]):
            #we make the key a tuple for consistency with later iterarions
            thimbles[index] = ProductThimbles(colored_state = (self.intersections_data[pt].segment+1,),
                                              hom_deg = self.intersections_data[pt].hom_deg,
                                              equ_deg = self.intersections_data[pt].equ_deg,
                                              order = (self.intersections_data[pt].order,),
                                              next_thimble_strand_number = 1,
                                              next_thimble_position = self.intersections_data[pt].segment+1,
                                              intersection_indices = (pt,))

        #for degree test
        indptr : cython.int
        indptr_end : cython.int
        for j in range(len(thimbles)):
                column_thimble = thimbles[j]
                
                indptr : cython.int = d_csc_current._indptrs()[j]
                indptr_end : cython.int = d_csc_current._indptrs()[j+1]
                while indptr != indptr_end:
                    i : cython.int = d_csc_current._indices()[indptr]
                    row_thimble = thimbles[i]
                    entry = d_csc_current._data()[indptr]
                    assert column_thimble.hom_deg == row_thimble.hom_deg + 1, "Cohomological degrees differ by an unexpected value"
                    #if row_thimble.equ_deg - column_thimble.equ_deg != entry.degree():
                    #    print("---", row_thimble.equ_deg, column_thimble.equ_deg, entry.degree(), entry.degree(reverse=True))
                    #else:
                    #    print("+++", row_thimble.equ_deg, column_thimble.equ_deg, entry.degree(), entry.degree(reverse=True))
                    assert row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(check_if_homogenous=True), row_thimble.equ_deg.__repr__() + " " + column_thimble.equ_deg.__repr__() + " " +  entry.degree().__repr__()
                    indptr += 1
        
        #on each step we add one more brane and correct the result
        #we call _current any things from the differential of all the branes before the step
        #and _next for the one-strand differential of the new brane
        for next_brane_number in range(1,len(self.branes)):
            thimbles_current = thimbles.copy()
            thimbles_next = {}
            for index,pt in zip(count(),self.branes[next_brane_number]):
                thimbles_next[index] = ProductThimbles(colored_state = (self.intersections_data[pt].segment+1,),
                                                       hom_deg = self.intersections_data[pt].hom_deg,
                                                       equ_deg = self.intersections_data[pt].equ_deg,
                                                       order = (self.intersections_data[pt].order,),
                                                       next_thimble_strand_number = 1,
                                                       next_thimble_position = self.intersections_data[pt].segment+1,
                                                       intersection_indices = (pt,))

            #the next_brane_diff is the differential for the next_brane_number-th brane
            d_csc_next = self.one_dimentional_differential(next_brane_number)

            #for degree test
            indptr : cython.int
            indptr_end : cython.int
            for j in range(len(thimbles_next)):
                    column_thimble = thimbles_next[j]
                    
                    indptr : cython.int = d_csc_next._indptrs()[j]
                    indptr_end : cython.int = d_csc_next._indptrs()[j+1]
                    while indptr != indptr_end:
                        i : cython.int = d_csc_next._indices()[indptr]
                        row_thimble = thimbles_next[i]
                        entry = d_csc_next._data()[indptr]
                        ind_c = column_thimble.intersection_indices
                        ind_r = row_thimble.intersection_indices
                        #print(self.intersections_data[ind_c[0]])
                        #print(self.intersections_data[ind_r[0]])
                        assert column_thimble.hom_deg == row_thimble.hom_deg + 1, "Cohomological degrees differ by an unexpected value"
                        #if row_thimble.equ_deg - column_thimble.equ_deg != entry.degree():
                        #    print("---", row_thimble.equ_deg, column_thimble.equ_deg, entry.degree(), entry.degree(reverse=True))
                        #else:
                        #    print("+++", row_thimble.equ_deg, column_thimble.equ_deg, entry.degree(), entry.degree(reverse=True))
                        assert row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(check_if_homogenous=True), row_thimble.equ_deg.__repr__() + " " + column_thimble.equ_deg.__repr__() + " " +  entry.degree().__repr__()
                        indptr += 1
                        
                
            thimbles = {}
            index = 0
            for index_current in thimbles_current:
                thimble_current = thimbles_current[index_current]
                point_orders_current_sorted = sorted(thimble_current.order)
                #print(point_orders_current_sorted)
                for index_next in thimbles_next:
                    thimble_next = thimbles_next[index_next]
                    order = thimble_current.order + thimble_next.order
                    hom_deg = thimble_current.hom_deg + thimble_next.hom_deg
                    intersection_indices = thimble_current.intersection_indices + thimble_next.intersection_indices
                    
                    #thimble_next.order and thimble_next.colored_state have one element each
                    order_next = thimble_next.order[0]
                    state_next = thimble_next.colored_state[0]

                    assert order_next not in point_orders_current_sorted

                    #finding the position of the next brane among the current branes
                    strands_to_the_left = bisect.bisect_left(point_orders_current_sorted, order_next)
                    #try:
                    #    strands_to_the_left = next(ind for ind,order in enumerate(point_orders_current_sorted) if order > order_next)
                    #except StopIteration:
                    #    strands_to_the_left = next_brane_number

                    #Now we know that the position of the last thimble is segment_next+i
                    #All points to the right of it have to add one to their strand number
                    position_next = state_next + strands_to_the_left
                    colored_state_list = list(thimble_current.colored_state)                    
                    for i in range(len(colored_state_list)):
                        if colored_state_list[i] >= position_next:
                            colored_state_list[i] += 1
                    colored_state_list.append(position_next)
                    colored_state = tuple(colored_state_list)
                    #the equivariant degree is the sum of the equivariant degrees of 1-d thimbles
                    #and the contributions for each pair
                    #the pair contributions inside of current are already taken into account in thimble_current.equ_deg
                    equ_deg = thimble_current.equ_deg + thimble_next.equ_deg
                    #the next thimble is 1-d, so only one point
                    pt_next = thimble_next.intersection_indices[0]
                    for pt_current in thimble_current.intersection_indices:
                        equ_deg += self.pairwise_equ_deg[pt_current,pt_next]

                    thimbles[index] = ProductThimbles(colored_state = colored_state,
                                                      hom_deg = hom_deg,
                                                      equ_deg = equ_deg,
                                                      order = order,
                                                      next_thimble_strand_number = strands_to_the_left + 1,
                                                      next_thimble_position = position_next,
                                                      intersection_indices = intersection_indices)
                    #print(thimble_current,thimble_next)
                    #print(index, " : ", thimbles[index])
                    index += 1


            #KRLW algebra on next_brane_number+1 strands
            KRLW = KRLWAlgebra(ZZ,u_name="u",hbar_name="h", n=self.n, k=next_brane_number+1, skip_computing_basis=True, warnings=True)

            #print(np.asarray(d_csc_next._data()))
            #print(np.asarray(d_csc_next._indices()))
            #print(np.asarray(d_csc_next._indptrs()))
            #print(d_csc_next._number_of_rows())
            
            #we are looking for the differential that is the tensor product of diff and next_brane_diff + corrections
            
            number_of_columns_current : cython.int = d_csc_current._number_of_rows()
            number_of_entries_current : cython.int = d_csc_current.nnz()
            number_of_columns_next : cython.int = d_csc_next._number_of_rows()
            number_of_entries_next : cython.int = d_csc_next.nnz()
            #each non-zero element in d_csc gives number_of_columns_next terms
            #each non-zero element in d_csc_next_brane gives number_of_columns_current terms
            #in the differential of form d \otimes 1 + (-1)^{...} 1 \otimes d
            #because of the homological gradings no terms are in the same matrix element
            number_of_entries : cython.int = number_of_entries_current*number_of_columns_next + number_of_entries_next*number_of_columns_current
            number_of_columns : cython.int = number_of_columns_current*number_of_columns_next
                
            d0_csc_data = np.empty(number_of_entries, dtype = "O")
            d0_csc_indices = np.zeros(number_of_entries, dtype = "intc")
            d0_csc_indptrs = np.zeros(number_of_columns+1, dtype = "intc")            
            
            j_current : cython.int
            j_next : cython.int
            #i_current : cython.int
            #i_next : cython.int
            indptr_current : cython.int
            indptr_next : cython.int
            indptr_end_current : cython.int
            indptr_end_next : cython.int
            entries_so_far : cython.int = 0
            for j_current in range(number_of_columns_current):
                #for signs later we remember cohomological degree
                cur_hom_degree = thimbles_current[j_current].hom_deg
                
                for j_next in range(number_of_columns_next):
                    #we write the column with the index (j_current,j_next), i.e. j_current*number_of_columns_next + j_next
                    #TODO: can make one case if we sort thimbles by cohomological degree first?

                    #column_thimble_current = thimbles_current[j_current]
                    #column_thimble_next = thimbles_next[j_next]
                    column_index = j_current*number_of_columns_next + j_next
                    column_thimble = thimbles[column_index]
                    column_next_position = column_thimble.next_thimble_position
                    column_colored_state = column_thimble.colored_state
                    column_state = frozenset(column_colored_state) #- frozenset((column_next_position,))
                    
                    
                    indptr_current = d_csc_current._indptrs()[j_current]
                    indptr_next = d_csc_next._indptrs()[j_next]
                    indptr_end_current = d_csc_current._indptrs()[j_current+1]
                    indptr_end_next = d_csc_next._indptrs()[j_next+1]

                    def add_d_times_one_term(indptr_current, entries_so_far):
                        row_index = d_csc_current._indices()[indptr_current]*number_of_columns_next + j_next
                        row_thimble = thimbles[row_index]
                        row_next_position = row_thimble.next_thimble_position
                        row_colored_state = row_thimble.colored_state
                        row_state = frozenset(row_colored_state) #- frozenset((row_next_position,))
                        bot_subset_of_strands = frozenset(range(1,next_brane_number+2)) - frozenset((row_thimble.next_thimble_strand_number,))

                        #print(column_index, column_thimble, row_index, row_thimble)
                        
                        entry_current = d_csc_current._data()[indptr_current]

                        entry = KRLW.zero()
                        for braid,coef in entry_current.monomial_coefficients().items():
                            #prepare the new braid by adding one more strand
                            mapping = {}
                            for t in column_colored_state:
                                if t != column_next_position:
                                    if t > column_next_position:
                                        t_cur = t-1
                                    else:
                                        t_cur = t
                                    b = braid.apply_as_permutation(t_cur,reverse=True)
                                    if b >= row_next_position:
                                        b += 1
                                    mapping[b] = t
                            d_times_one_word = word_by_extending_permutation(mapping,N=self.n+next_brane_number+1)
                            #now add with dots, corrected to be on more strands
                            for exp,scalar in coef.iterator_exp_coeff():
                                uhpart = exp[:2]
                                epart = exp[2:next_brane_number+2]
                                xpart = braid.x
                                dots_dict = KRLW.dots_from_dots_on_less_strands(subset_of_strands=bot_subset_of_strands,
                                                                                uhpart=uhpart, 
                                                                                epart=epart, 
                                                                                xpart=xpart)
                                for xpart_ETuple,central_elem in dots_dict.items():
                                    entry += scalar*central_elem*KRLW.KRLWmonomial(top=column_state, bot=row_state,
                                                                                   x=tuple(xpart_ETuple), word=d_times_one_word)

                        #print([thimbles_current[j_current],thimbles_next[j_next]])
                        #print([thimbles_current[d_csc_current._indices()[indptr_current]],thimbles_next[j_next]])
                        #ind_c = column_thimble.intersection_indices
                        #ind_r = row_thimble.intersection_indices
                        #print([self.intersections_data[i] for i in ind_c])
                        #print(self.pairwise_equ_deg[ind_c[0],ind_c[1]])
                        #print([self.intersections_data[i] for i in ind_r])
                        #print(self.pairwise_equ_deg[ind_r[0],ind_r[1]])
                        #print(entry_current.degree(), entry.degree())
                        #print(entry_current, entry)
                        assert column_thimble.hom_deg == row_thimble.hom_deg + 1, "Cohomological degrees differ by an unexpected value"
                        ##assert row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(check_if_homogenous=True), row_thimble.equ_deg.__repr__() + " " + column_thimble.equ_deg.__repr__() + " " +  entry.degree().__repr__()
                        
                        if row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(check_if_homogenous=True):
                            d0_csc_indices[entries_so_far] = row_index
                            d0_csc_data[entries_so_far] = entry
                            return entries_so_far+1
                        else:
                            # d0_csc_data[entries_so_far] = KRLW.zero()
                            return entries_so_far
                        

                    def add_one_times_d_term(indptr_next, entries_so_far):
                        row_index = j_current*number_of_columns_next + d_csc_next._indices()[indptr_next]
                        row_thimble = thimbles[row_index]
                        row_next_position = row_thimble.next_thimble_position
                        row_colored_state = row_thimble.colored_state
                        row_state = frozenset(row_colored_state)# - frozenset((row_next_position,))
                        bot_subset_of_strands = frozenset((row_thimble.next_thimble_strand_number,))
                        
                        entry_next = d_csc_next._data()[indptr_next]

                        entry = KRLW.zero()
                        for braid,coef in entry_next.monomial_coefficients().items():
                            #prepare the new braid by adding one more strand
                            mapping = {row_next_position:column_next_position}
                            one_times_d_word = word_by_extending_permutation(mapping,N=self.n+next_brane_number+1)
                            #now add with dots, corrected to be on more strands
                            for exp,scalar in coef.iterator_exp_coeff():
                                uhpart = exp[:2]
                                #for one strand the length of epart is 1 and no xpart
                                epart = exp[2:3]
                                dots_dict = KRLW.dots_from_dots_on_less_strands(subset_of_strands=bot_subset_of_strands,
                                                                                uhpart=uhpart, 
                                                                                epart=epart, 
                                                                                xpart=None)
                                for xpart_ETuple,central_elem in dots_dict.items():
                                    #print(braid, row_next_position, column_next_position)
                                    entry += scalar*central_elem*KRLW.KRLWmonomial(top=column_state, bot=row_state, 
                                                                                   x=tuple(xpart_ETuple), word=one_times_d_word)

                        #print([thimbles_current[j_current],thimbles_next[j_next]])
                        #print([thimbles_current[j_current],thimbles_next[d_csc_next._indices()[indptr_next]]])
                        #ind_c = column_thimble.intersection_indices
                        #ind_r = row_thimble.intersection_indices
                        #print([self.intersections_data[i] for i in ind_c])
                        #print(self.pairwise_equ_deg[ind_c[0],ind_c[1]])
                        #print([self.intersections_data[i] for i in ind_r])
                        #print(self.pairwise_equ_deg[ind_r[0],ind_r[1]])
                        #print(entry.degree())
                        #print(entry)
                        assert column_thimble.hom_deg == row_thimble.hom_deg + 1, "Cohomological degrees differ by an unexpected value"
                        ##assert row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(check_if_homogenous=True), row_thimble.equ_deg.__repr__() + " " + column_thimble.equ_deg.__repr__() + " " +  entry.degree().__repr__()
                        if row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(check_if_homogenous=True):
                            d0_csc_indices[entries_so_far] = row_index
                            if cur_hom_degree % 2 == 0:
                                d0_csc_data[entries_so_far] = entry
                            else:
                                d0_csc_data[entries_so_far] = -entry
                            return entries_so_far + 1
                        else:
                            # d0_csc_data[entries_so_far] = KRLW.zero()
                            return entries_so_far    
                        
                    
                    #for this range of row indices in d_csc_current only d \otimes 1 contributes
                    while indptr_current != indptr_end_current:
                        if d_csc_current._indices()[indptr_current] < j_current:
                            entries_so_far = add_d_times_one_term(indptr_current, entries_so_far)
                            # entries_so_far += 1
                            indptr_current += 1
                        else:
                            break
                        
                    #terms of form (-1)^... 1 \times d
                    while indptr_next != indptr_end_next:
                        if d_csc_next._indices()[indptr_next] < j_next:
                            # d0_csc_indices[entries_so_far] = j_current*number_of_columns_next + d_csc_next._indices()[indptr_next]
                            entries_so_far = add_one_times_d_term(indptr_next, entries_so_far)
                            # entries_so_far += 1
                            indptr_next += 1
                        else:
                            break
                    
                    #one term from d \otimes 1
                    if indptr_current != indptr_end_current:
                        if d_csc_current._indices()[indptr_current] == j_current:
                            entries_so_far = add_d_times_one_term(indptr_current, entries_so_far)
                            # entries_so_far += 1
                            indptr_current += 1
                            
                    #remaining terms of form (-1)^... 1 \times d
                    while indptr_next != indptr_end_next:
                        # d0_csc_indices[entries_so_far] = j_current*number_of_columns_next + d_csc_next._indices()[indptr_next]
                        entries_so_far = add_one_times_d_term(indptr_next, entries_so_far)
                        # entries_so_far += 1
                        indptr_next += 1
                    
                    #now row indices in d_csc_current are >j_current, again only d \otimes 1 contributes
                    while indptr_current != indptr_end_current:
                        entries_so_far = add_d_times_one_term(indptr_current, entries_so_far)
                        # entries_so_far += 1
                        indptr_current += 1
                    
                    d0_csc_indptrs[column_index + 1] = entries_so_far

            d0_csc_indices = d0_csc_indices[:entries_so_far]
            d0_csc_data = d0_csc_data[:entries_so_far]

            for i in range(len(d0_csc_data)):
                if d0_csc_data[i].is_zero():
                    print("Zero element in the possible corrections matrix:", i)
            d_csc = CSC_Mat(data = d0_csc_data, indices = d0_csc_indices, indptrs = d0_csc_indptrs, number_of_rows = number_of_columns)

            M = multiply(d_csc.to_csr(),d_csc)
#            for x in M._data():
#                print(x)
#            print(np.asarray())
            S = Solver(KRLW)
            S.set_d0(d_csc)
            S.check_d0()
#            d1_csc, number_of_variables = self.differential_u_corrections(thimbles=thimbles, order=1, k=next_brane_number+1)
#            for order in count(start = 1):
#                print("----------Correcting order {} in u for the product of the first {} branes----------".format(order,next_brane_number+1))
#                d1_csc, number_of_variables = self.differential_u_corrections(thimbles=thimbles, order=order, k=next_brane_number+1)
#                S.set_d1(d1_csc, number_of_variables = number_of_variables)
#                multiplier = S.KRLW_algebra().u**order
#                S.make_corrections(multiplier = multiplier, order = order, graded_type = "u^order*h^0")
#                if S.d0_squares_to_zero_mod(ideal = KRLW.center.ideal([KRLW.hbar])):
#                    break

#            S.check_d0()

#            print(S.d0().as_an_array())
            d1_csc, number_of_variables = self.differential_u_corrections(thimbles=thimbles, order=None, k=next_brane_number+1, d_csc=S.d0())
            S.set_d1(d1_csc, number_of_variables = number_of_variables)
            for order in range(1,4):
                print("----------Correcting order {} in h for the product of the first {} branes----------".format(order,next_brane_number+1))
                multiplier = (S.KRLW_algebra().hbar * S.KRLW_algebra().u)**order
                S.make_corrections(multiplier = multiplier, order = order, graded_type = "h^order")
                if S.d0_squares_to_zero():
                    break
            
        self.differential = S.d0()
        self.thimbles = thimbles

    #@cython.ccall
    def find_differential_matrix(self, domain_indices : list, codomain_indices : list, R : PrincipalIdealDomain) -> Matrix_sparse:
        d_dict = {}
        for ind_j,j in enumerate(domain_indices):
            for ind_i,i in enumerate(codomain_indices):
                #temporary fix: our d0 is transposed to fix the multiplication order difference from Elise's version
                elem = self.differential[j,i]
                if elem != None:
                    
                    no_terms_found = True
                    for braid, coef in elem.monomial_coefficients().items():
                        if braid.word == () and braid.x == (0,) * self.k:
                            #print(braid,coef)
                            for exp, scalar in coef.iterator_exp_coeff():
                                #if the term has no dots
                                if exp[2:] == ETuple((0,) * self.k):
                                    d_dict[ind_i,ind_j] = scalar
                    #                #checking if the terms is a power of u*h
                    #                assert exp[0] == exp[1], "Terms in d are not in Field[u*h]"
                    #                #checking if we have only one power of u*h
                    #                assert no_terms_found, "Terms in d are not monomials in u*h!"
                    #                #if self.KRLW.center.base_ring() != ZZ:
                    #                #    #make entires integral, check that the entries are close to integers
                    #                #    scalar_int = ZZ(np.rint(scalar))
                    #                #    if abs(scalar - scalar_int)>self.tolerance():
                    #                #        print("Warning: element far from integral:",elem)
                    #                    #assert abs(scalar - scalar_int)<self.tolerance(), "A scalar {} in a matrix element {} of d are not close enough to integers!".format(scalar, elem)
                    #                #else:
                    #                #    scalar_int = scalar
                    #                #d_dict[ind_i,ind_j] = scalar_int
                    #                d_dict[ind_i,ind_j] = scalar
                    #                no_terms_found = False

        d_mat = matrix(R, ncols = len(domain_indices), nrows = len(codomain_indices), entries = d_dict, sparse = True)
        return d_mat

    def part_of_graded_component(
            self, 
            relevant_thimbles, 
            current_hom_deg, 
            current_equ_deg, 
            i,
        ):
            if i == len(relevant_thimbles):
                return False
            if relevant_thimbles[i][1].hom_deg != current_hom_deg:
                return False
            if relevant_thimbles[i][1].equ_deg != current_equ_deg:
                return False
            return True

    def find_cohomology(self, R : PrincipalIdealDomain):
        """
        Working over R that is a PID. 
        [works on fields and integers, modules over other PIDs not fully implemented in Sage yet]
        Returns a Poincare Polynomial if R is a field 
        and a dictionary {degrees:invariant factors} if R is the ring of integers
        """
        
        #{2,4,6,...} in Elise's convention
        #{2,5,8,...} in our convention
        relevant_uncolored_state = frozenset(
            2+3*i
            for i in range(self.k)
        )

        relevant_thimbles = [
            (i, st)
            for i, st in self.thimbles.items() 
            if st.uncolored_state() == relevant_uncolored_state
        ]

        print("The number of relevant thimbles:", len(relevant_thimbles))

        #here we will use that it our convention differential *preserves* the equivariant degree
        #and increases the homological degree by 1.
        relevant_thimbles.sort(key=(lambda x: (x[1].equ_deg, x[1].hom_deg)))
            
#        degrees_and_indices = np.zeros(shape = (len(relevant_states),), 
#                                       dtype=[('equ_deg', 'intc'), ('hom_deg', 'intc'), ('index', 'intc')]
#                                      )
    
#        for a,i in zip(relevant_states,count()):
#            degrees_and_indices[i] = (self.equivariant_degrees[a], self.cohomological_degrees[a], a)

#        print(degrees_and_indices)

        #here we will use that it our convention differential *preserves* the equivariant degree
        #and increases the homological degree by 1.
#        degrees_and_indices.sort(order=('equ_deg','hom_deg'))
        
        d_prev : Matrix_sparse
        d_next : Matrix_sparse
        C1_indices : list
        C2_indices : list
        C3_indices : list

        #we will find all triples C1, C2, C3 that they enter as
        #C1->C2->C3 in the complex, possibly with zero C1 or C3.
        #Then we find the maps -> which we call d_prev and d_next
        #and compute the homology at C2.
        if R.is_field():            
            LauPoly = LaurentPolynomialRing(ZZ,2, ["t","q"])
            t = LauPoly("t")
            q = LauPoly("q")
            PoincarePolynomial = LauPoly(0)
        else:
            Homology = {}
        i = 0
        C3_indices = []
        while i != len(relevant_thimbles):
            #if the chain did not break by ending as ->0 on the previous step
            if C3_indices:
                C1_indices = C2_indices
                d_prev = d_next.__copy__()
                current_hom_deg += 1
                C2_indices = C3_indices

                C3_indices = []
                while self.part_of_graded_component(relevant_thimbles, current_hom_deg+1, current_equ_deg, i):
                    C3_indices.append(relevant_thimbles[i][0])
                    i += 1

                d_next = self.find_differential_matrix(C2_indices, C3_indices, R)

                assert (d_next*d_prev).is_zero()
                #print(C1_indices,"->",C2_indices,"->",C3_indices)
                #print(d_prev,d_next)
                #print("-----")

                if R.is_field():
                    PoincarePolynomial += (d_next.right_nullity()-d_prev.rank()) * t**current_hom_deg * q**current_equ_deg
                else: 
                    homology_group = d_next.right_kernel() / d_prev.column_module()
                    invariants = homology_group.invariants()
                    if invariants:
                        Homology[current_hom_deg,current_equ_deg] = [R.quotient(inv*R) for inv in invariants]

            else:
                C1_indices = []
                C2_indices = []
                #degrees of C2
                current_hom_deg = relevant_thimbles[i][1].hom_deg
                current_equ_deg = relevant_thimbles[i][1].equ_deg

                while self.part_of_graded_component(relevant_thimbles, current_hom_deg, current_equ_deg, i):
                    C2_indices.append(relevant_thimbles[i][0])
                    i += 1

                #we already know C3_indices = [] from if
                while self.part_of_graded_component(relevant_thimbles, current_hom_deg+1, current_equ_deg, i):
                    C3_indices.append(relevant_thimbles[i][0])
                    i += 1

                d_next = self.find_differential_matrix(C2_indices, C3_indices, R)
                #print(C1_indices,"->",C2_indices,"->",C3_indices)
                #print(d_next)
                #print("-----")

                if R.is_field():
                    PoincarePolynomial += d_next.right_nullity() * t**current_hom_deg * q**current_equ_deg
                
                ##over a PID a submodule of a free module is free
                else:
                    homology_group = d_next.right_kernel()
                    invariants = [0]*homology_group.rank()
                    if invariants:
                        Homology[current_hom_deg,current_equ_deg] = [R.quotient(inv*R) for inv in invariants]
                        
        if R.is_field():
            return PoincarePolynomial
        else:
            return Homology
