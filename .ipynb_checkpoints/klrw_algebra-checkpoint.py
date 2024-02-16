import cython

from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.categories.finite_dimensional_algebras_with_basis import FiniteDimensionalAlgebrasWithBasis
from sage.structure.parent import Set_generic
from sage.structure.unique_representation import UniqueRepresentation
from sage.categories.rings import Rings
from sage.categories.action import Action

from itertools import product
from typing import NamedTuple
from collections.abc import Iterable
import operator
from sage.rings.polynomial.polydict import ETuple

#from functools import cache
from sage.misc.lazy_attribute import lazy_attribute
from sage.misc.cachefunc import cached_method
from sage.misc.misc_c import prod

from klrw_braid import KLRWbraid, KLRWbraid_set
from framed_dynkin import NodeInFramedQuiver, FramedDynkinDiagram_with_dimensions, KLRWUpstairsDotsAlgebra#, KLRWDownstairsDotsAlgebra
from bimodule_monoid import LeftFreeBimoduleMonoid

from klrw_endomorphism import KLRWEndomorphismAlgebra, LeftKLRWEndomorphismAction, RightKLRWEndomorphismAction


class KLRWElement(IndexedFreeModuleElement):
    def degree(self, check_if_homogeneous=False):
        #zero elements return None as degree
        degree = None
        for braid,coeff in self:
            if not coeff.is_zero():
                braid_degree = self.braid_degree(self.parent().quiver, braid)
                coeff_degree = coeff.parent().element_degree(coeff, check_if_homogeneous=check_if_homogeneous)
                term_degree = braid_degree + coeff_degree
                    
                if not check_if_homogeneous:
                    return term_degree
                elif degree is None:
                    degree = term_degree
                elif degree != term_degree:
                    raise ValueError("The KLRW element is not homogeneous!")
                        
        return degree

    
    @staticmethod
    def braid_degree(quiver, braid):
        degree = 0
        current_state = braid.right_state()
            
        for i in reversed(braid.word()):
            degree += -quiver.scalar_product_of_simple_roots(current_state[i-1],current_state[i])
            current_state = current_state.act_by_s(i)
        
        return degree

    #TODO:rewrite
    #def check(self):
    #    for braid,__ in self.monomial_coefficients().items():
    #        try:
    #            braid.check()
    #        except:
    #            print(self.monomial_coefficients().items())
    #            raise ValueError("Braid {} in a KLRW element is incorrect".format(braid))
        
    #TODO: add checks that all braids are correct


class RightDotAction(Action):
    #@cached_method
    def _act_(self, p, x : IndexedFreeModuleElement) -> IndexedFreeModuleElement:
        return self.codomain().linear_combination((self._act_on_bases_(exp_tuple,braid), coeff * left_poly )
                                                  for exp_tuple,coeff in p.iterator_exp_coeff(as_ETuples=True)
                                                  for braid,left_poly in x )
        

    
    @cached_method
    def _act_on_bases_(self, p_exp_tuple : ETuple, braid : KLRWbraid):
        #print(p_exp_tuple, braid)
        return self.codomain().sum(self._act_on_bases_iter_(p_exp_tuple,braid))

        
    def _act_on_bases_iter_(self, p_exp_tuple : ETuple, braid : KLRWbraid):
        #idempotents commute with dots
        #print("--",p_exp_tuple, braid,"--")
        if len(braid.word()) == 0:
            #print("-1-", braid)
            yield self.codomain()._from_dict({braid: self.actor().monomial(*p_exp_tuple)})
        else:
            last_letter = braid.word()[-1]
            c1,c2 = braid.intersection_colors(-1)
            if c1 == c2:
                index_of_dots_on_left_strand = self.actor().variables[c1,braid.right_state().index_among_same_color(last_letter-1)].position
                index_of_special_coefficient = self.actor().variables[c1].position
                #print(self.G.gen(index_of_special_coefficient))
                #terms where crossing stays
                coeff_of_crossing = self._coeff_of_crossing_(p_exp_tuple, index_of_dots_on_left_strand)
                for exp_tuple,coeff in coeff_of_crossing.iterator_exp_coeff(as_ETuples=True):
                    for term in self._act_on_bases_iter_(exp_tuple, braid[:-1]):
                        for br, poly in term:
                            #print("-2-")
                            #we use :meth:right_multiply_by_s, to bring to lexmin form
                            yield coeff*poly*self.codomain().right_multiply_by_s(br, braid[-1])
                #terms where crossing goes away
                #print([x for x in self._coeff_of_correction_(p_exp_tuple, index_of_dots_on_left_strand, index_of_special_coefficient)])
                for poly in self._coeff_of_correction_(p_exp_tuple, index_of_dots_on_left_strand, index_of_special_coefficient):
                    #print(poly)
                    for exp_tuple,coeff in poly.iterator_exp_coeff(as_ETuples=True):
                        #print("-3-", braid[:-1])
                        yield coeff*self._act_on_bases_(exp_tuple, braid[:-1])
    
            else:  
                for term in self._act_on_bases_iter_(p_exp_tuple, braid[:-1]):
                    for br, poly in term:
                        #print("-4-")
                        yield poly*self.codomain().right_multiply_by_s(br, braid[-1])

    
    def _coeff_of_crossing_(self, p_exp_tuple : ETuple, i : int):
        '''
        We will write X*dot = _coeff_of_crossing_*X + _coeff_of_correction_*||
        Here X is a simple crossing and || is a smoothed version
        i and i+1 are the indices of variables corresponding to the dots on the crossing strands
        '''
        new_exp = list(p_exp_tuple)
        new_exp[i+1],new_exp[i] = new_exp[i],new_exp[i+1]
        #return self.G.monomial(*tuple(new_exp))
        #print(self.actor().monomial(*new_exp))
        return self.actor().monomial(*new_exp)

    
    def _coeff_of_correction_(self, p_exp_tuple : ETuple, i : int, j : int) -> Iterable:
        '''
        We will write X*dot = _coeff_of_crossing_*X + r*_coeff_of_correction_*||
        Here X is a simple crossing and || is a smoothed version
        i and i+1 are the indices of variables corresponding to the dots on the crossing strands
        j is the index of r
        '''
        #returns None if two degrees coincide [no corrections]
        if p_exp_tuple[i] != p_exp_tuple[i+1]:
            new_exp = list(p_exp_tuple)
            
            if p_exp_tuple[i] > p_exp_tuple[i+1]:
                end_degree_at_i = new_exp[i+1]-1
                new_exp[i] += -1
                delta = 1
            else:
                end_degree_at_i = new_exp[i+1]
                new_exp[i+1] += -1
                delta = -1
            
            #implicitly multiplying by t_{...}
            new_exp[j] += 1
            
            while new_exp[i]!=end_degree_at_i:
                #we do tuple() to make a copy and plug it in later into :meth:G.monomial
                yield self.actor().base()(delta)*self.actor().monomial(*tuple(new_exp))
                new_exp[i] += -delta
                new_exp[i+1] += delta            

        

        
class KLRWAlgebra(LeftFreeBimoduleMonoid):
    Element = KLRWElement
    
    def __init__(self, base_R, quiver : FramedDynkinDiagram_with_dimensions, warnings=False, downstairs=False, **prefixes):
        """
        Makes a KLRW algebra for a quiver.
        base_R is the base ring.
        if warnings, then extra warnings can be printed
        """
        self.warnings = warnings
        self.quiver = quiver#.copy()
        self.KLRWBraid = KLRWbraid_set(quiver, state_on_right = True)
        #if downstairs:
            #too slow.
            #Do operations upstairs and return the reduced elements?
        #    dots_algebra = KLRWDownstairsDotsAlgebra(base_R, quiver)
        #else:
        dots_algebra = KLRWUpstairsDotsAlgebra(base_R, quiver, **prefixes)
        category=FiniteDimensionalAlgebrasWithBasis(dots_algebra)
        super().__init__(R = dots_algebra, category=category)
        #can add element_class=KLRWElement as a parameter

    
    @lazy_attribute
    def ideal_of_symmetric_dots(self):
        return list(self.base().symmetric_dots_gens())*self.base()

    
    def modulo_symmetric_dots(self, element):
        return element.reduce(self.ideal_of_symmetric_dots)

    
    #We don't call it :meth:one to avoid unintentional coercion from the ring of dots/scalars. It works, but it too slow.
    @cached_method
    def _one_(self):
        return self.sum_of_terms((self.KLRWBraid._element_constructor_(state),self.base().one())
                                 for state in self.KLRWBraid.KLRWstate_set)
            
    
    def gens(self):
        #dots and other coefficients
        for state in self.KLRWBraid.KLRWstate_set:
            for scalar in self.base().gens():
                yield self._from_dict({self.KLRWBraid._element_constructor_(state):scalar})
        #simple crossings
        yield from self.gens_over_dots()

    
    def gens_over_dots(self):
        #simple crossings
        for state in self.KLRWBraid.KLRWstate_set:
            for i in range(1,len(state)):
                if not state[i-1].is_framing() and not state[i].is_framing():
                    yield self._from_dict({self.KLRWBraid._element_constructor_(state,(i,)):self.base().one()})

            
    def center_gens(self):
        yield from self.base().center_gens()

    
    def basis_in_dots_modulo_center(self):
        yield from self.base().basis_modulo_symmetric_dots()

    
    def basis_over_dots_and_center(self):
        '''
        Returns a basis [as a left module] over the ring generated by dots and parameters.
        '''
        for braid in self.KLRWBraid:
            yield self.monomial(braid)

    
    def basis_over_center(self):
        for braid,dot_poly in product(self.KLRWBraid,self.basis_in_dots_modulo_center()):
            yield self._from_dict({braid:dot_poly})

        
    def _get_action_(self, other, op, self_on_left):
        is_left = not self_on_left
        if op == operator.mul:
            if self_on_left == True:
                if self.base() == other:
                #if self.base().has_coerce_map_from(other):
                    return RightDotAction(other, self, is_left = is_left, op = operator.mul)
                if self.endomorphisms == other:
                    return RightKLRWEndomorphismAction(other, self, is_left = is_left, op = operator.mul)
            else:
                #if self.endomorphisms.has_coerce_map_from(other):
                if self.endomorphisms == other:
                    return LeftKLRWEndomorphismAction(other, self, is_left = is_left, op = operator.mul)

    
    @lazy_attribute
    def endomorphisms(self):
        number_of_moving_strands = 0
        for node,dim in self.quiver.dimensions().items():
            if not node.is_framing():
                number_of_moving_strands += dim
        return KLRWEndomorphismAlgebra(self,number_of_moving_strands)
                
    
    def clear_cache(self, verbose=True):
        """
        Clears cache from all @cached_method's
        """
        import inspect
        from sage.misc.cachefunc import CachedMethodCaller
        for name,m in self.__dict__.items():   
            if isinstance(m,CachedMethodCaller):
                m.clear_cache()
                if verbose:
                    print("Cleared cache of",name)
                
            
        
#    def inject_parameters(self, scope=None, verbose=True):
#        """
#        Defines globally u,hbar-variables
#        """
#        vs = ["u","h"]
#        gs = [self.u, self.hbar]
#        if scope is None:
#            scope = globals()
#        if verbose:
#            print("Defining %s" % (', '.join(vs)))
#        for v, g in zip(vs, gs):
#            scope[v] = g

    
    
    #slightly modified coercion defined in CombinatorialFreeModule
    def _coerce_map_from_(self, A):
        if isinstance(A, KLRWAlgebra):
            if self.center.has_coerce_map_from(A.center):
                return True
            
        return super()._coerce_map_from_(A)
    
    
    def _element_constructor_(self, x):    
        if isinstance(x.parent(), KLRWAlgebra):
            if self.center.has_coerce_map_from(x.parent().center):
                d = {basis: self.center(coef) 
                     for basis, coef in x._monomial_coefficients.items() 
                     if not self.center(coef).is_zero()
                    }
                return self._from_dict(d)
        
        return super()._element_constructor_(x)
    
    
    
#    @cached_method
#    def _dots_of_weight_as_tuples_(self, weight, k):
#        """
#        Makes a list of all pairs of tuples of length k that in _ring_of_dots_ represent a basis in the space of dots of given weight.
#        k is the number of strands
#        """
#        assert k>0
#        if k==1:
#            return [((weight,),(0,))]
#        else:
#            #e_k has weight k
#            max_power_of_right_e = weight // k
#            #return a list of tuples (epart,xpart) via recursion
#            return [((*epart,p),(q,*xpart)) 
#                    for p in range(max_power_of_right_e+1) 
#                    for q in range(min(weight - k*p + 1, k)) 
#                    for epart,xpart in self._dots_of_weight_as_tuples_(weight-k*p-q, k-1)]
    
#    @cached_method
#    def dots_of_weight(self, weight):
#        """
#        Basis in the algebra of dots of given weight.
#        """
#        basis = []
#        for epart,xpart in self._dots_of_weight_as_tuples_(weight, self.k):
#            basis.append(self._ring_of_dots_.monomial(*(epart+xpart)))
#        return basis

#    @cached_method
#    def basis_in_graded_component(self, right_state:frozenset, left_state:frozenset, equivariant_degree=None, number_of_dots=None, reverse=False):
#        """
#        Basis in the subspace of given weight or number of dots with given right and left states.
#        If lefth equivariant degrees and number of dots are given only elements satisfying lefth criteria are listed
#        """
#        if self.warnings:
#            assert len(right_state) == len(left_state)
#            assert len(right_state) == self.k
#        
#        #at least one should be given
#        assert (equivariant_degree is not None) or (number_of_dots is not None)
#            
#        basis = []
#        if self.k == 1:
#            #recording the only integers in the sets
#            right_state_int = tuple(right_state)[0]
#            left_state_int = tuple(left_state)[0]
#            braid = KLRWbraid_one_strand(right_state=right_state_int,
#                                         left_state=left_state_int,
#                                         xpart=self.no_x_part,
#                                         N=self.k+self.n
#                                        )
#            
#            braid_degree = 0
#            if reverse:
#                if right_state_int > left_state_int:
#                    braid_degree = right_state_int - left_state_int
#            else:
#                if right_state_int < left_state_int:
#                    braid_degree = left_state_int - right_state_int
#                    
#            if equivariant_degree is not None:
#                current_number_of_dots = equivariant_degree - braid_degree
#                if number_of_dots is not None:
#                    #in this case we don't have any matching braids, so we just make the number of dots negative
#                    if number_of_dots != equivariant_degree - braid_degree:
#                        current_number_of_dots = -1
#            #in this case, by assertion, number_of_dots was given
#            else:
#                current_number_of_dots = number_of_dots
#                
#            if current_number_of_dots >= 0:
#                #(0,0,...) stays for zero powers of hbar and u in the center 
#                #the coefficient is e1^number_of_dots [equivalently, x1^number_of_dots]
#                basis += [self.term(index=braid, coeff=self.center.monomial(0,0,current_number_of_dots))] 
#            
#        else:
#            for p in Permutations(self.k):
#                braid = KLRWbraid_from_states_and_permutation(right_state=right_state,
#                                                              left_state=left_state,
#                                                              perm=p,
#                                                              xpart=self.no_x_part,
#                                                              N=self.k+self.n)
#                
#                #if equivariant degree is given instead of the number of dots
#                if equivariant_degree is not None:
#                    braid_degree = braid.degree(reverse=reverse)
#                    current_number_of_dots = equivariant_degree - braid_degree
#                    if number_of_dots is not None:
#                        #in this case we don't have any matching braids, so we just make the number of dots negative
#                        if number_of_dots != equivariant_degree - braid_degree:
#                            current_number_of_dots = -1
#                #in this case, by assertion, number_of_dots was given
#                else:
#                    current_number_of_dots = number_of_dots
#                    
#                if current_number_of_dots >= 0:
#                    #(0,0,...) stays for zero powers of hbar and u in the center 
#                    basis += [self.term(index=braid._replace(x=xpart), coeff=self.center.monomial(0,0,*epart)) 
#                              for epart,xpart in self._dots_of_weight_as_tuples_(current_number_of_dots, self.k)]
#            
#        return basis
    
    #@cached_method
    def KLRWmonomial(self, state, word=()):
        """
        Returns a monomial corresponding to the data
        If  no word is given, it returns the idempotent corresponding to state
        """
        #checks?
        return self.monomial(self.KLRWBraid._element_constructor_(state=state,word=word))

        
    def idempotent(self, state):
        """
        Returns an idempotent corresponding to the state
        """
        return self.KLRWmonomial(state)

    
    #@cached_method
    def right_multiply_by_s(self, m, i):
        """
        Multiplies by a braid representing an elementary transposition. The transposition is on the right.
        m is a KLRW braid.
        i is the index of an elementary transposition
        Main ingredient of computing the product
        """
        #print(m,i)
        return self.sum(self._right_action_of_s_iter_(m,i))
        

    def _right_action_of_s_iter_(self, m, i):
        #print(m,i)
        #looking at (i-1)-th and i-th elements in the sequence
        old_right_state = m.right_state()
        left_color, right_color = old_right_state[i-1], old_right_state[i]
        #print(left_color, right_color)
        
        new_right_state = old_right_state.act_by_s(i)
        intersection_index = m.find_intersection(i-1, i, right_state=True)
        #braid_after_intersection is the braid is the braid after the intersection of i-th and j-th strands
        #after the existing intersection if they intersect
        #after the new intersection if they don't intersect
        if intersection_index == -1:
            #if don't intersect
            new_intersection_index, new_intersection_position = m.position_for_new_s_from_right(i)
            #new_intersection_index -= 1
            braid_after_intersection = m[new_intersection_index:]
        else:
            new_intersection_index = intersection_index + 1
            braid_after_intersection = m[intersection_index+1:]

        #print("braid_after_intersection",braid_after_intersection)

        current_state = new_right_state#braid_after_intersection.right_state()
        left_strand_position_among_same_color = old_right_state.index_among_same_color(i-1)
        word_after_intersection = braid_after_intersection.word()
        current_ind = len(word_after_intersection)-1
        #print("current_ind",current_ind)
        for ind,color in braid_after_intersection.intersections_with_given_strand(i-1):
            #print("----",ind,color,left_color,right_color)
            if color == left_color:
                d_ij = -self.quiver[left_color,right_color]
                #print("++", d_ij, braid_after_intersection, i, ind)
                if d_ij>0:
                    current_state = self.KLRWBraid.find_state_on_other_side(state = current_state, 
                                                                            word = word_after_intersection[ind+1:current_ind+1], 
                                                                            reverse = True)
                    current_ind = ind
                    current_ind_in_original_word = ind + new_intersection_index
                    #d_ji = -self.quiver[right_color,left_color] 
                    #print(left_color, right_color, color)
                    #print(m,i)
                    r_i = self.base().variables[left_color].monomial
                    t_ij = self.base().variables[left_color,right_color].monomial
                    #obsolete, but in the else case we can mltiply faster because there are no dots
                    lower_part = (-r_i*t_ij)*self.KLRWmonomial(state=current_state, word = m.word()[:current_ind_in_original_word-1])
                    if d_ij>1:
                        x_left = self.base().variables[left_color,left_strand_position_among_same_color-1].monomial
                        x_right = self.base().variables[left_color,left_strand_position_among_same_color].monomial
                        lower_part = lower_part*sum((x_left**l)*(x_right**(d_ij-1-l)) for l in range(d_ij))
                        
                    #print("2", intersection_index, m.word(), current_ind_in_original_word, word_after_intersection, ind)
                    yield lower_part*self.KLRWmonomial(state = new_right_state, word = word_after_intersection[ind+1:])
                #since we crossed another strand with the same color, the position among the same color changes
                left_strand_position_among_same_color -= 1

        #print(current_state)
        current_state = self.KLRWBraid.find_state_on_other_side(state = current_state, 
                                                                word = word_after_intersection[:current_ind+1], 
                                                                reverse = True)
        #print("current_ind",current_ind)

        if intersection_index == -1:
            index, position = m.position_for_new_s_from_right(i)
            new_word = m.word()[:index] + (position,) + m.word()[index:]
            #print("3", new_right_state, new_word)
            assert len(new_word) == len(m.word())+1
            yield self.KLRWmonomial(state = new_right_state, word = new_word)
        #if do intersect
        else:
            if left_color != right_color:
                d_ij = -self.quiver[left_color,right_color]
                if d_ij>=0:
                    t_ij = self.base().variables[left_color,right_color].monomial
                    lower_part = self.KLRWmonomial(state=current_state, word = m.word()[:intersection_index])
                    if d_ij>0:
                        d_ji = -self.quiver[right_color,left_color]
                        t_ji = self.base().variables[right_color,left_color].monomial
                        x_left = self.base().variables[left_color,left_strand_position_among_same_color].monomial
                        right_strand_position_among_same_color = current_state.index_among_same_color(m.word()[intersection_index]-1)
                        #try:
                        x_right = self.base().variables[right_color,right_strand_position_among_same_color].monomial
                        #except:
                        #    print(m,i)
                        #    print(current_state,m.word(),intersection_index,right_color,right_strand_position_among_same_color)
                        lower_part = lower_part*(t_ij*x_left**(d_ij)+t_ji*x_right**(d_ji))
                        #print("4", lower_part, word_after_intersection, new_right_state)
                    else:
                        lower_part = t_ij*lower_part
                        #print("5")
                    #assert len(m.word()[:intersection_index]) + len(word_after_intersection) == len(m.word())-1
                    yield lower_part*self.KLRWmonomial(state = new_right_state, word = word_after_intersection)
        
            
    @cached_method
    def product_on_basis(self, left, right):
        """
        Computes the product of two KLRW braids
        This method is called by _mul_
        left an right are KLRWbraids
        """
        #print("++", left, right)
        if left.right_state() != right.left_state():
            if self.warnings:
                print("states don't match!",left.right_state(), right.left_state())
                print(left, right)
            return self.zero()

        if not right.word():
            #print("1", left, right)
            return self.monomial(left)

        #if the left word is empty
        if not left.word():
            #print("2", left, right)
            return self.monomial(right)
        
        #if right has no dots and has non-trivial word
        #one can show that if right.word() is lexmin shortest, then right[1:].word() is also lexmin shortest
        return self.right_multiply_by_s(left,right[0]) * self.monomial(right[1:])   
             

    def _repr_term(self, monomial):
        """
        Single-underscore method for string representation of basis elements
        """

        s = "E_"+monomial.left_state().__repr__()
        for i in monomial.word():
            s += "*s{}".format(i)

        if monomial.word():
            s += "*E_"+monomial.right_state().__repr__()
        
        return s
       
        
    def _repr_(self):
        """
        Single-underscore method for string representation
        """
        return "KLRW algebra"
    