from itertools import product
from functools import cache

from sage.rings.integer_ring import IntegerRing
from sage.matrix.matrix_space import MatrixSpace

from framed_dynkin import NodeInFramedQuiverWithMarks
from klrw_algebra import KLRWAlgebra

def maps_to(index):
    """
    Generator returning (sign,next_index)
    Sign is the Koszul sign,
    next_index is the index where one of the zeroes is replaced by a one
    """
    iter = (x for x in range(len(index)) if index[x] == 0)
    for i in iter:
        left_part = index[:i]
        right_part = index[i+1:]
        degree = sum(right_part)
        sign = 1 if degree%2 == 0 else -1
        yield sign, left_part + (1,) + right_part

def braid_from_marked_state(left_state, right_state):
    '''
    We assume that both states have marks that make each strand type unique
    '''
    #make reduced word by finding elements of the Lehmer cocode
    rw = []
    permutation = {left_index:right_state.index(mark) for left_index,mark in enumerate(left_state)}
    for left_index,mark in enumerate(left_state):
        right_index = permutation[left_index]
        cocode_element = sum(1 for i in range(left_index) if permutation[i]>right_index)
        piece = [j + 1 for j in range(left_index-1,left_index-cocode_element-1,-1)]
        rw += piece
            
    word = tuple(rw)
    return word

@cache
def flatten_index(index):
    result = 0
    for i in index:
        result = 2*result + i
    return result

@cache
def marked_state_by_index(ind, left_framing, left_sequence, right_sequence, right_framing):
    '''
    TODO: Make independent of global variables
    '''
    state = [left_sequence[x] for x in range(len(left_sequence)-1,-1,-1) if ind[x] == 0]
    state += [left_framing]
    state += [left_sequence[x] for x in range(len(left_sequence)) if ind[x] == 1]
    state += [right_sequence[x] for x in range(len(right_sequence)-1,-1,-1) if ind[x+len(left_sequence)] == 0]
    state += [right_framing]
    state += [right_sequence[x] for x in range(len(right_sequence)) if ind[x+len(left_sequence)] == 1]
    return tuple(state)

@cache
def unmark(iterable):
    return iterable.__class__(i.unmark() for i in iterable)

def find_differential(DD, left_framing, right_framing, sequence, dots_on_left):
    '''
    TODO: Make independent of global variables
    '''
    DD[left_framing] += 1
    DD[right_framing] += 1
    for node in sequence:
        DD[node] += 1

    print("Framed quiver:")
    print(DD)

    KLRW = KLRWAlgebra(IntegerRing(),DD, warnings = True)
    Mat = MatrixSpace(KLRW, 2**len(sequence), sparse=True)
    Braid = KLRW.KLRWBraid
    State = Braid.KLRWstate_set
    State.enable_checks()
    d = Mat.matrix()

    #mark moving strands to distinguish them
    sequence = tuple(NodeInFramedQuiverWithMarks(node, i) for i,node in enumerate(sequence))
    
    left_framing = NodeInFramedQuiverWithMarks(left_framing, 0)
    right_framing = NodeInFramedQuiverWithMarks(right_framing, 1)
    
    left_sequence = sequence[:dots_on_left]
    right_sequence = sequence[dots_on_left:]
    
    I = product(*([range(2)]*len(sequence)))
    for i in I:
        marked_state = marked_state_by_index(i, left_framing, left_sequence, right_sequence, right_framing)
        state = State._element_constructor_(unmark(marked_state))
        for sign, new_index in maps_to(i):
            new_marked_state = marked_state_by_index(new_index, left_framing, left_sequence, right_sequence, right_framing)
            word = braid_from_marked_state(new_marked_state, marked_state)
            new_state = State._element_constructor_(unmark(new_marked_state))
            braid = Braid._element_constructor_(state, word)
            assert braid.left_state() == new_state
            d[flatten_index(new_index), flatten_index(i)] = sign*KLRW.monomial(braid)

    return d