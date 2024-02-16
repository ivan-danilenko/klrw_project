from sage.structure.element_wrapper import ElementWrapper
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.parent import Set_generic
from sage.categories.sets_cat import Sets
from sage.misc.cachefunc import cached_method
from sage.combinat.permutation import Permutations

from framed_dynkin import FramedDynkinDiagram_with_dimensions

class KLRWstate(ElementWrapper):
    wrapped_class = tuple
    __lt__ = ElementWrapper._lt_by_value
    #slots save memory by not allowing other
    __slots__ = ()

    
    def __init__(self, parent, iterable):
        super().__init__(parent, iterable)
        if self.parent()._check:
            self.parent().check_element(self)

    
    def __iter__(self):
        return iter(self.value)

    
    def __len__(self):
        return len(self.value)

    
#    def __set__(self, instance, value):
#        raise TypeError("States are immutable.")

    
    def __setattr__(cls, name, value):
        '''
        Make sure elements behave like immutable ones:
        One can't change the value once it was created.
        '''
        if name == "value":
            raise AttributeError("Cannot modify .value")
        else:
            return type.__setattr__(cls, name, value)

    
    #def __getitem__(self, index):
    #    return self.value.__getitem__(index)

    
    def __getitem__(self, key):
        if isinstance(key, slice):
            #we modify the state, but don't want to recompute the dimensions. We make a set without dimension checks.
            generic_state_set = self.parent().__class__()
            return generic_state_set._element_constructor_(self.value[key])
        else:        
            return self.value[key]

    
    def count(self, *args, **kwargs):
        return self.value.count(*args, **kwargs)

    
    def act_by_s(self, i):
        return self.parent().act_by_s(i, self.value)

    
    def index_among_same_color(self, i):
        '''
        Here index starts with 1.
        '''
        color = self.value[i]
        return self.value[:i].count(color)+1
    


class KLRWstate_set(UniqueRepresentation, Set_generic):
    Element = KLRWstate

    
    def _element_constructor_(self, iterable):
        return self._element_constructor_from_tuple_(tuple(iterable))

    
    @cached_method
    def _element_constructor_from_tuple_(self, tuple):
        return self.element_class(self, tuple)

        
#    def __init__(self, **dimensions):
    def __init__(self, quiver = None):
        '''
        Dimentions have to be given as a tuple of (node,dimension)
        '''
        self._category = Sets().Facade()
        self._check = False

        if quiver is not None:
            self._dimensions = quiver.dimensions()

            #self._dimensions = {n:d for n,d in dimensions.values()}
            self._total_number_of_strands = sum(d for d in self._dimensions.values())
            #do some checks
            for node,d in self._dimensions.items():
                try:
                    dim = int(d)
                except ValueError:
                    raise ValueError("Node dimensions must be integers.")
                assert dim>=0, "Node dimensions must be non-negative."

    
    def __iter__(self):
        assert hasattr(self, "_dimensions"), "Dimensions have to be defined"
        lowest_state = []
        for node,dim in self._dimensions.items():
            lowest_state += [node]*dim
        for state in Permutations(lowest_state):
            yield self._element_constructor_(state)

    
    def enable_checks(self):
        assert hasattr(self, "_dimensions"), "Dimensions have to be defined"
        self._check = True

    
    def disable_checks(self):
        assert hasattr(self, "_dimensions"), "Dimensions have to be defined"
        self._check = False

    
    def dimensions(self, copy=False):
        assert hasattr(self, "_dimensions"), "Dimensions have to be defined"
        if copy:
            return self._dimensions.copy()
        else:
            return self._dimensions

    
    def check_element(self, element):
        assert hasattr(self, "_dimensions"), "Dimensions have to be defined"
        #print(self._dimensions, element)
        for i in element:
            #for j in self._dimensions:
            #    print(i,j, i is j, i == j, i.__class__, j.__class__)
            assert i in self._dimensions, "Unknown index used: {}".format(i)
        for node,dim in self._dimensions.items():
            assert dim == element.count(node), "The node dimension for {} is incorrect.".format(node)

    
    @cached_method
    def act_by_s(self, i, element):
        '''
        i must be from 1 to number_of_strands-1
        '''
        new_element = list(element)
        #we have a shift because the elements in a tuple are numbered by 1, ..., number_of_strands
        new_element[i-1] = element[i]
        new_element[i] = element[i-1]
        return self._element_constructor_(new_element)


#def KLRWstates_from_framed_quiver(framed_quiver : FramedDynkinDiagram_with_dimensions):
#    dims = {node.__repr__():(node,dim) for node,dim in framed_quiver.dimensions().items()}
#    return KLRWstate_set(**dims)