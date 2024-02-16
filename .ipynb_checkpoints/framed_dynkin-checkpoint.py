from typing import NamedTuple
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from sage.combinat.root_system.dynkin_diagram import DynkinDiagram_class
from sage.combinat.root_system.cartan_type import CartanType_abstract
#from sage.structure.factory import UniqueFactory
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomialRing_libsingular as PolynomialRing
from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular as Polynomial
from sage.rings.polynomial.term_order import TermOrder
#from sage.rings.quotient_ring import QuotientRing_generic
#from sage.rings.quotient_ring_element import QuotientRingElement
from sage.rings.ring import CommutativeRing

from sage.misc.cachefunc import cached_method
from sage.misc.misc_c import prod

from itertools import product


class NodeInFramedQuiver(NamedTuple):
    node : object
    framing : bool = False

    
    def is_framing(self):
        return self.framing

    
    def __repr__(self):
        if self.framing:
            return "W"+self.node.__repr__()
        else:
            return "V"+self.node.__repr__()

    
    def make_framing(self):
        return self._replace(framing = True)


class NodeInFramedQuiverWithMarks(NamedTuple):
    unmarked_node : NodeInFramedQuiver
    mark : object

    
    def unmark(self):
        return self.unmarked_node

    
    def __repr__(self):
        return self.unmarked_node.__repr__() + "_" + self.mark.__repr__()


#class Node(UniqueFactory):
#    def create_key(self, node, is_framing = False):
#        return (node, is_framing)
#    def create_object(self, version, key, **extra_args):
#        # We ignore version and extra_args
#        return NodeInFramedQuiver(key)

        
class FramedDynkinDiagram_class(DynkinDiagram_class):
    def __init__(self, t=None, **options):
        assert isinstance(t,CartanType_abstract)
        #add a property framing = False to all labels
        relabel_dict = {v:NodeInFramedQuiver(v) for v in t.index_set()}
        t_name = t.__repr__()
        t = t.relabel(relabel_dict)
        
        super().__init__(t.dynkin_diagram())
        #add framing vertices
        for v in t.index_set():
            w = v.make_framing()
            self.add_vertex(w)
            #in this convention marks are negative the off-diagonal coefficients
            self.add_edge(v,w,1)
            self.add_edge(w,v,1)

    
    def _repr_(self):        
        ct = self.cartan_type()
        is_linear = ct.type() == "A" or ct.type() == "B" or ct.type() == "C"
        is_linear &= ct.is_finite()
        if is_linear:
            framed_node = "□"
            node = "◯"
            n = ct.rank()
            if n == 0:
                return ""
            result = "".join("{!s:4}".format(v.make_framing()) for v in ct.index_set()) + "\n"
            result += "   ".join(framed_node for i in range(n)) + "\n"
            result += "   ".join("|" for i in range(n)) + "\n"
            if ct.type() == "A" or n == 1:
                result += "---".join(node for i in range(n)) + "\n"
            elif ct.type() == "B":
                result += "---".join(node for i in range(1,n)) + "=>=" + node + "\n"
            elif ct.type() == "C":
                result += "---".join(node for i in range(1,n)) + "=<=" + node + "\n"
            result += "".join("{!s:4}".format(v) for v in ct.index_set())
        else:
            result = "Framed Quiver"
            
        return result #+"\n"+"Framed Quiver"

    
    @cached_method
    def scalar_product_of_simple_roots(self, i, j):
        ct = self.cartan_type()
        sym = ct.symmetrizer()
        if not i.is_framing():
            return self[i,j]*sym[i]
        else:
            return self[j,i]*sym[j]

    
    def is_simply_laced(self):
        return self.cartan_type().is_simply_laced()

    
    def non_framing_nodes(self):
        return self.cartan_type().index_set()

    
    def inject_nodes(self, scope=None, verbose=True):
        """
        Defines globally nodes
        """
        vs = [v.__repr__() for v in self.vertices()]
        gs = [v for v in self.vertices()]
        if scope is None:
            scope = globals()
        if verbose:
            print("Defining %s" % (', '.join(vs)))
        for v, g in zip(vs, gs):
            scope[v] = g

            
    def KLRW_vertex_param_names(self, vertex_prefix = "r"):
        vertex_names = {}
        edge_names = {}
        #iterating over non-framing vertices
        for v in self.non_framing_nodes():
            vertex_names[v] = vertex_prefix + "_{}".format(v.node)

        return vertex_names

        
    def KLRW_edge_param_names(self, edge_prefix = "t", framing_prefix = "u"):
        edge_names = {}
        for n1,n2,_ in self.edges():
            #should we add second set of framing variables? Or just rescale?
            if n2.is_framing():
                assert n1.node == n2.node
                edge_names[n1,n2] = framing_prefix + "_{}".format(n1.node)
            else:
                if not n1.is_framing():
                    edge_names[n1,n2] = edge_prefix + "_{}_{}".format(n1.node,n2.node)

        return edge_names
        

    def KLRW_special_edge_param_names(self, special_edge_prefix = "s"):
        #these are half of the squares of lenghts normalized to have squared length 2 for short roots
        ct = self.cartan_type()
        half_lengths_squared = ct.symmetrizer()
        special_edge_names = {}
        for n1,n2,w in ct.dynkin_diagram().edges():
            print(n1,n2,w)
            if not n1.is_framing() and not n2.is_framing():
                if w > 1:
                    a = half_lengths_squared[n1]
                    b = half_lengths_squared[n2]
                    p = 1
                    
                    while w-a*p>=0:
                        q = (w-a*p)/b
                        special_edge_names[n1,n2] = (p,q)
                        print(a,"*",p, " + ", b, "*", q, " = ", w )
                        p += 1
                        
                #special_edge_names[n1,n2] = special_edge_prefix + "_{}_{}".format(n1.node,n2.node)
                #???

        return special_edge_names
        

class FramedDynkinDiagram_with_dimensions(FramedDynkinDiagram_class):
    def __init__(self, t=None, **options):
        assert isinstance(t,CartanType_abstract)
        super().__init__(t)
        self._dimensions = defaultdict(int)

    
    def __hash__(self):
        '''
        We need hash to be able pass instances of this class to __init__'s of UniqueRepresentation classes [e.g. KLRW algebra]
        '''
        return hash((self.dimensions_list(),self.cartan_type()))
#        import hashlib
#        import json
#
#        dhash = hashlib.md5()
        # We need to sort arguments so {'a': 1, 'b': 2} is
        # the same as {'b': 2, 'a': 1}
#        encoded = json.dumps(({"a":1}, {"b":2}), sort_keys=True).encode()
#        dhash.update(encoded)
#        return dhash.hexdigest()

    
    def __getitem__(self, key):
        '''
        If key is a NodeInFramedQuiver returns dimensions in the framing
        Otherwise returns an element of the Cartan matrix.
        '''
        if isinstance(key, NodeInFramedQuiver):
            return self._dimensions[key]
        else:
            return super().__getitem__(key)

    
    def __setitem__(self, key, dim):
        '''
        A more convenient way to set dimentions
        '''
        if isinstance(key, NodeInFramedQuiver):
            self._dimensions[key] = dim
        if isinstance(key, tuple):
            node = NodeInFramedQuiver(*key)
        else:
            node = NodeInFramedQuiver(key)
        assert node in self.vertices()
        self._dimensions[node] = dim
        

    def dimensions_list(self):
        return tuple(sorted(self._dimensions.items()))
        

    def dimensions(self, copy = False):
        if copy:
            return self._dimensions.copy()
        else:
            return self._dimensions

    
    def get_dim(self, *index):
        """
        Gets the dimension of a node
        Allows to access
        """
        if len(index) == 1:
            if isinstance(index[0],NodeInFramedQuiver):
                return self._dimensions[index]
            node = NodeInFramedQuiver(index[0], False)
            if node in self.cartan_type().index_set():
                return self._dimensions[node]
        if len(index) == 2:
            assert isinstance(index[1],bool)
            return self._dimensions[NodeInFramedQuiver(*index)]
            
        raise ValueError("Incorrect index")
        
        
    def set_dim(self, dim, index, is_framing = False):
        node = NodeInFramedQuiver(index,is_framing)
        assert node in self.vertices()
        self._dimensions[node] = dim
        
    
    def _repr_(self):
        ct = self.cartan_type()
        is_linear = ct.type() == "A" or ct.type() == "B" or ct.type() == "C"
        is_linear &= ct.is_finite()
        if is_linear:
            result = "".join("{!s:4}".format(self._dimensions[v]) for v in self.vertices() if v.is_framing()) + "\n"
            result += super()._repr_() + "\n"
            result += "".join("{!s:4}".format(self._dimensions[v]) for v in self.vertices() if  not v.is_framing())
        else:
            result = "Framed Quiver of type {}{}".format(ct.type(), ct.rank())
        return result

    
    def KLRW_dots_names(self, dots_prefix = "x"):
        dots_names = {}
        #iterating over non-framing vertices
        for v in self.cartan_type().index_set():
            for k in range(self._dimensions[v]):
                dots_names[v,k+1] = dots_prefix + "_{}_{}".format(v.node,k+1)
        return dots_names

    
    def KLRW_deformations_names(self, deformations_prefix = "z"):
        deformations_names = {}
        for v in self.cartan_type().index_set():
            w = v.make_framing()
            for k in range(self._dimensions[w]):
                deformations_names[w,k+1] = deformations_prefix + "_{}_{}".format(w.node,k+1)
        return deformations_names

    
    def names(self, no_deformations = True, **prefixes):
        '''
        Returns a dictionary {label: name}
        '''
        #We need the dictionary to keep the order of elements
        #In Python 3.7 and higher it is guaranteed
        #We also use |= from Python 3.9 and higher
        names = {}
        names |= self.KLRW_vertex_param_names(prefixes.get("vertex_prefix", "r"))
        names |= self.KLRW_edge_param_names(prefixes.get("edge_prefix", "t"), prefixes.get("framing_prefix","u"))
        names |= self.KLRW_dots_names(prefixes.get("dots_prefix", "x"))
        if not no_deformations:
            names |= self.KLRW_deformations_names(prefixes.get("deformations_prefix","z"))

        return names
        

@dataclass(slots=True)
class QuiverParameter:
    #index : object
    name : str
    position : int
    monomial : object

    
    def __repr__(self):
        return self.monomial.__repr__()
        #if self.name:
        #    return self.name
        #else:
        #    return self.monomial.__repr__()


#class VariablesDict(dict):
#    '''
#    Introduces default values for variables
#    '''
#    variables_dictionary : dict
#    default_edge_value : object
#    default_edge_value

    
def is_a_dot_index(index):
    if not isinstance(index, NodeInFramedQuiver):
        if not isinstance(index[1], NodeInFramedQuiver):
            if not index[0].is_framing():
                return True
    return False
    

#TODO: add an abstract class KLRWDotsAlgebra(CoomutativeRing, UniqueRep), 
#inherit upstairs and downstairs versions
class KLRWDotsAlgebra(CommutativeRing, UniqueRepresentation):
    #have to define self.variables
    def assign_default_values(self, quiver, no_deformations = True):
        #TODO: add default parameters' options, maybe by changing self.variables to a custom dictionary
        #second set of framing variables rescaled to 1
        for n1,n2 in product(quiver.vertices(),quiver.vertices()):
            if (n1,n2) not in self.variables:
                self.variables[n1,n2] = QuiverParameter(None, None, self.one())
        if no_deformations:
            for key in quiver.KLRW_deformations_names():
                self.variables[key] = QuiverParameter(None, None, self.zero())

    
    def center_gens(self):
        elementary_symmetric_polynomials = {}
        for index, variable in self.variables.items():
            if not isinstance(index, NodeInFramedQuiver):
                if not isinstance(index[1], NodeInFramedQuiver):
                    if not index[0].is_framing():
                        continue
            if variable.name is not None:
                yield variable.monomial
        yield from self.symmetric_dots_gens()

    
    def symmetric_dots_gens(self):
        raise NotImplementedError()

    
    def basis_modulo_symmetric_dots(self):
        variables = []
        degree_ranges = []
        for index, variable in self.variables.items():
            if is_a_dot_index(index):
                degree_ranges.append(range(index[1]))
                variables.append(variable.monomial)
        degree_iterator = product(*degree_ranges)
        for degree in degree_iterator:
            yield prod(var**deg for var,deg in zip(variables,degree))
            

class KLRWUpstairsDotsAlgebra(PolynomialRing, KLRWDotsAlgebra):
    def __init__(self, base_ring, quiver : FramedDynkinDiagram_with_dimensions, no_deformations = True, order = "degrevlex", **prefixes):
        names = quiver.names(no_deformations = no_deformations, **prefixes)
        names_list = [name for _,name in names.items()]

        super().__init__(base_ring, len(names_list), names_list, order)

        self.degrees = [quiver.scalar_product_of_simple_roots(index[0],index[0]) if is_a_dot_index(index) else 0 for index,name in names.items() ]
        #type of variables is uniquely determined by its index data
        self.variables = {index:QuiverParameter(name, position, self(name)) for position,(index,name) in enumerate(names.items())}
        
        self.assign_default_values(quiver, no_deformations = no_deformations)

    
    def symmetric_dots_gens(self):
        elementary_symmetric_polynomials = {}
        for index, variable in self.variables.items():
            if is_a_dot_index(index):
                if index[0] in elementary_symmetric_polynomials:
                    e_list = elementary_symmetric_polynomials[index[0]]
                    #add the top elementary symmetric polynomial
                    e_list.append(variable.monomial*e_list[-1])
                    #modify intermediate symmetric polynomials
                    for i in range(len(e_list)-2,0,-1):
                        e_list[i]+= variable.monomial*e_list[i-1]
                    #modify the first elementary symmetric polynomial
                    e_list[0] += variable.monomial
                else:
                    #for one variable there is only the first elementary symmetric polynomial
                    elementary_symmetric_polynomials[index[0]] = [variable.monomial]
                continue
        for _,e_list in elementary_symmetric_polynomials.items():
            yield from e_list

    
    #libsingular wrapper in Sage does not allow to change elements to a wrapper, so we define degree in the parent
    def element_degree(self, element, check_if_homogeneous=False):
        #zero elements return None as degree
        degree = None
        variables_degrees = self.degrees
        for exp,scalar in element.iterator_exp_coeff():
            if not scalar.is_zero():
                term_degree = sum(deg*power for deg,power in zip(variables_degrees,exp))
                
                if not check_if_homogeneous:
                    return term_degree
                elif degree is None:
                    degree = term_degree
                elif degree != term_degree:
                    raise ValueError("The dots are not homogeneous!")
                    
        return degree

        
#class PolynomialQuotientRingElement(QuotientRingElement):
#    def __getattr__(self, item):
#        #print(self.__rep)
#        return getattr(self.lift(), item)

        
#    def _richcmp_(left, right):
#        print(self,other)
#        parent = self.parent()
#        I = parent.defining_ideal()
#        return self.reduce(I) == other.reduce(I)

#class PolynomialQuotientRingElement(Polynomial):
#    def reduce(self):
#        parent = self.parent()
#        I = parent.defining_ideal()
#        return parent._element_constructor_(I.reduce(self))


#class KLRWDownstairsDotsAlgebra(QuotientRing_generic, KLRWDotsAlgebra):
#    Element = PolynomialQuotientRingElement
#    def __init__(self, base_ring, quiver : FramedDynkinDiagram_with_dimensions, no_deformations = True, order = "degrevlex", **prefixes):
#        upstairs_algebra = KLRWUpstairsDotsAlgebra(base_ring = base_ring, quiver = quiver, no_deformations = no_deformations, order = order, **prefixes)
#        I = list(upstairs_algebra.symmetric_dots_gens())*upstairs_algebra
#        names = [quiver_parameter.name for _,quiver_parameter in upstairs_algebra.variables.items() if quiver_parameter.name is not None]
#        super().__init__(upstairs_algebra, I, names = names)
        #type of variables is uniquely determined by its index data
#        self.variables = {index:QuiverParameter(name = quiver_parameter.name, position = quiver_parameter.position, monomial = self(quiver_parameter.monomial)) for index,quiver_parameter in upstairs_algebra.variables.items()}

        
#    def monomial(self, *args, **kwdargs):
#        return self(self.ambient().monomial(*args, **kwdargs))

        
#    def symmetric_dots_gens(self):
#        '''
#        Empty generator
#        '''
#        yield from ()