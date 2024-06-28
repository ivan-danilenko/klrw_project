from typing import NamedTuple
from collections import defaultdict
from dataclasses import dataclass, InitVar, replace
from types import MappingProxyType
from itertools import product

import sage.all

from sage.combinat.root_system.dynkin_diagram import DynkinDiagram_class
from sage.combinat.root_system.cartan_type import CartanType_abstract
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.polynomial.multi_polynomial_libsingular import (
    MPolynomialRing_libsingular as PolynomialRing_sing,
)
from sage.rings.polynomial.laurent_polynomial_ring import (
    LaurentPolynomialRing_mpair as LaurentPolynomialRing,
)
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
# from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial_mpair
from sage.rings.polynomial.polydict import ETuple
from sage.rings.ring import CommutativeRing, Ring
from sage.rings.integer_ring import ZZ
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.structure.sage_object import SageObject

from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.misc.misc_c import prod


class NodeInFramedQuiver(NamedTuple):
    node: object
    framing: bool = False

    def is_framing(self):
        return self.framing

    def __repr__(self):
        if self.framing:
            return "W" + self.node.__repr__()
        else:
            return "V" + self.node.__repr__()

    def make_framing(self):
        return self._replace(framing=True)


class NonOrientedEdge(frozenset):
    def __new__(cls, vertex1, vertex2):
        assert isinstance(vertex1, NodeInFramedQuiver)
        assert isinstance(vertex2, NodeInFramedQuiver)
        return super(NonOrientedEdge, cls).__new__(cls, frozenset((vertex1, vertex2)))

    def vertices(self):
        if len(self) == 2:
            vertex1, vertex2 = self
        if len(self) == 1:
            (vertex1,) = self
            vertex2 = vertex1
        return vertex1, vertex2

    def __repr__(self):
        vertex1, vertex2 = self.vertices()
        return repr(vertex1) + "--" + repr(vertex2)

    def __reduce__(self):
        vertex1, vertex2 = self.vertices()
        return (
            self.__class__,
            (vertex1, vertex2),
        )


class NodeInFramedQuiverWithMarks(NamedTuple):
    unmarked_node: NodeInFramedQuiver
    mark: object

    def unmark(self):
        return self.unmarked_node

    def __repr__(self):
        return self.unmarked_node.__repr__() + "_" + self.mark.__repr__()


class FramedDynkinDiagram_class(UniqueRepresentation, DynkinDiagram_class):
    def __init__(self, t=None, **options):
        assert isinstance(t, CartanType_abstract)
        # add a property framing = False to all labels
        relabel_dict = {v: NodeInFramedQuiver(v) for v in t.index_set()}
        t = t.relabel(relabel_dict)

        super().__init__(t.dynkin_diagram())
        # add framing vertices
        for v in t.index_set():
            w = v.make_framing()
            self.add_vertex(w)
            # in this convention marks are negative the off-diagonal coefficients
            self.add_edge(v, w, 1)
            self.add_edge(w, v, 1)

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
            result = (
                "".join("{!s:4}".format(v.make_framing()) for v in ct.index_set())
                + "\n"
            )
            result += "   ".join(framed_node for i in range(n)) + "\n"
            result += "   ".join("|" for i in range(n)) + "\n"
            if ct.type() == "A" or n == 1:
                result += "---".join(node for i in range(n)) + "\n"
            elif ct.type() == "B":
                result += "---".join(node for i in range(1, n)) + "=>=" + node + "\n"
            elif ct.type() == "C":
                result += "---".join(node for i in range(1, n)) + "=<=" + node + "\n"
            result += "".join("{!s:4}".format(v) for v in ct.index_set())
        else:
            result = "Framed Quiver"

        return result  # +"\n"+"Framed Quiver"

    @cached_method
    def scalar_product_of_simple_roots(self, i, j):
        ct = self.cartan_type()
        sym = ct.symmetrizer()
        if not i.is_framing():
            return self[i, j] * sym[i]
        else:
            return self[j, i] * sym[j]

    def is_simply_laced(self):
        return self.cartan_type().is_simply_laced()

    @cached_method
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
            print("Defining %s" % (", ".join(vs)))
        for v, g in zip(vs, gs):
            scope[v] = g

    def KLRW_vertex_param_names(self, vertex_prefix="r"):
        vertex_names = {}
        # iterating over non-framing vertices
        for v in self.non_framing_nodes():
            vertex_names[VertexVariableIndex(v)] = vertex_prefix + "_{}".format(v.node)

        return vertex_names

    def KLRW_edge_param_names(self, edge_prefix="t", framing_prefix="u"):
        edge_names = {}
        for n1, n2, _ in self.edges():
            # should we add second set of framing variables? Or just rescale?
            if n2.is_framing():
                assert n1.node == n2.node
                edge_names[EdgeVariableIndex(n1, n2)] = framing_prefix + "_{}".format(
                    n1.node
                )
            else:
                if not n1.is_framing():
                    edge_names[EdgeVariableIndex(n1, n2)] = (
                        edge_prefix + "_{}_{}".format(n1.node, n2.node)
                    )

        return edge_names

    def KLRW_special_edge_param_names(self, special_edge_prefix="s"):
        # these are half of the squares of lenghts normalized
        # to have squared length 2 for short roots
        ct = self.cartan_type()
        half_lengths_squared = ct.symmetrizer()
        special_edge_names = {}
        for n1, n2, w in ct.dynkin_diagram().edges():
            print(n1, n2, w)
            if not n1.is_framing() and not n2.is_framing():
                if w > 1:
                    a = half_lengths_squared[n1]
                    b = half_lengths_squared[n2]
                    p = 1

                    while w - a * p >= 0:
                        q = (w - a * p) / b
                        special_edge_names[n1, n2] = (p, q)
                        print(a, "*", p, " + ", b, "*", q, " = ", w)
                        p += 1

        raise NotImplementedError("Special parameters need more work")
        return special_edge_names


@dataclass(frozen=True, repr=False)
class FramedDynkinDiagram_with_dimensions(SageObject):
    ct: InitVar[CartanType_abstract | None] = None
    quiver: FramedDynkinDiagram_class | None = None
    dimensions_dict: dict | defaultdict | MappingProxyType | None = None

    def __post_init__(self, ct: CartanType_abstract | None):
        if ct is not None:
            assert isinstance(ct, CartanType_abstract), (
                "Don't know how to make a Framed Dynkin Diagram"
                + " with dimensions from {}".format(ct.__class__)
            )
            if self.quiver is not None:
                assert self.quiver.cartan_type() == ct
            else:
                # bypass protection in frozen=True to change the entry
                super().__setattr__(
                    "quiver",
                    FramedDynkinDiagram_class(ct),
                )
        else:
            assert self.quiver is not None, "Quiver must be given"

        if self.dimensions_dict is None:
            # bypass protection in frozen=True to change the entry
            super().__setattr__(
                "dimensions_dict",
                defaultdict(int),
            )
        if isinstance(self.dimensions_dict, list):
            # bypass protection in frozen=True to change the entry
            super().__setattr__(
                "dimensions_dict",
                {key: value for key, value in self.dimensions_dict},
            )

    def __hash__(self):
        """
        We need hash to be able pass instances of this class to __init__'s
        of UniqueRepresentation classes [e.g. KLRW algebra]
        """
        return hash((self.dimensions_list(), self.quiver.cartan_type()))

    #    def __getattr__(self, name):
    #        return getattr(self.quiver, name)

    def __getitem__(self, key):
        """
        If key is a NodeInFramedQuiver returns dimensions in the framing
        Otherwise returns an element of the Cartan matrix.
        """
        if isinstance(key, NodeInFramedQuiver):
            return self.dimensions_dict[key]
        else:
            return self.quiver[key]

    def __setitem__(self, key, dim):
        """
        A more convenient way to set dimentions
        """
        if isinstance(key, NodeInFramedQuiver):
            self.dimensions_dict[key] = dim
        if isinstance(key, tuple):
            node = NodeInFramedQuiver(*key)
        else:
            node = NodeInFramedQuiver(key)
        assert node in self.quiver.vertices()
        self.dimensions_dict[node] = dim

    def dimensions_list(self):
        return tuple(sorted(self.dimensions_dict.items()))

    def dimensions(self, copy=False):
        if copy:
            return self.dimensions_dict.copy()
        else:
            return self.dimensions_dict

    def get_dim(self, *index):
        """
        Gets the dimension of a node
        Allows to access
        """
        if len(index) == 1:
            if isinstance(index[0], NodeInFramedQuiver):
                return self.dimensions_dict[index]
            node = NodeInFramedQuiver(index[0], False)
            if node in self.quiver.non_framing_nodes():
                return self.dimensions_dict[node]
        if len(index) == 2:
            assert isinstance(index[1], bool)
            return self.dimensions_dict[NodeInFramedQuiver(*index)]

        raise ValueError("Incorrect index")

    def set_dim(self, dim, index, is_framing=False):
        node = NodeInFramedQuiver(index, is_framing)
        assert node in self.quiver.vertices()
        self.dimensions_dict[node] = dim

    def scalar_product_of_simple_roots(self, *args, **kwargs):
        return self.quiver.scalar_product_of_simple_roots(*args, **kwargs)

    def is_simply_laced(self, *args, **kwargs):
        return self.quiver.is_simply_laced(*args, **kwargs)

    def non_framing_nodes(self, *args, **kwargs):
        return self.quiver.non_framing_nodes(*args, **kwargs)

    def inject_nodes(self, *args, **kwargs):
        self.quiver.inject_nodes(*args, **kwargs)

    def KLRW_vertex_param_names(self, *args, **kwargs):
        return self.quiver.KLRW_vertex_param_names(*args, **kwargs)

    def KLRW_edge_param_names(self, *args, **kwargs):
        return self.quiver.KLRW_edge_param_names(*args, **kwargs)

    def KLRW_special_edge_param_names(self, *args, **kwargs):
        return self.quiver(*args, **kwargs)

    def _repr_(self):
        ct = self.quiver.cartan_type()
        is_linear = ct.type() == "A" or ct.type() == "B" or ct.type() == "C"
        is_linear &= ct.is_finite()
        if is_linear:
            result = (
                "".join(
                    "{!s:4}".format(self.dimensions_dict[v])
                    for v in self.quiver.vertices()
                    if v.is_framing()
                )
                + "\n"
            )
            result += self.quiver._repr_() + "\n"
            result += "".join(
                "{!s:4}".format(self.dimensions_dict[v])
                for v in self.quiver.vertices()
                if not v.is_framing()
            )
        else:
            result = "Framed Quiver of type {}{}".format(ct.type(), ct.rank())
        return result

    def KLRW_dots_names(self, dots_prefix="x"):
        dots_names = {}
        # iterating over non-framing vertices
        for v in self.quiver.non_framing_nodes():
            for k in range(self.dimensions_dict[v]):
                dots_names[DotVariableIndex(v, k + 1)] = dots_prefix + "_{}_{}".format(
                    v.node, k + 1
                )
        return dots_names

    def KLRW_deformations_names(self, deformations_prefix="z"):
        deformations_names = {}
        for v in self.quiver.non_framing_nodes():
            w = v.make_framing()
            for k in range(self.dimensions_dict[w]):
                deformations_names[DotVariableIndex(w, k + 1)] = (
                    deformations_prefix + "_{}_{}".format(w.node, k + 1)
                )
        return deformations_names

    def names(
        self,
        no_deformations=True,
        no_vertex_parameters=False,
        no_edge_parameters=False,
        **prefixes,
    ):
        """
        Returns a two dictionaries {label: name}
        The first one is for non-dot variables
        The second one is for dot variables
        """
        # We need the dictionary to keep the order of elements
        # In Python 3.7 and higher it is guaranteed
        # We also use |= from Python 3.9 and higher
        nondot_names = {}
        if not no_vertex_parameters:
            nondot_names |= self.KLRW_vertex_param_names(
                prefixes.get("vertex_prefix", "r")
            )
        if not no_edge_parameters:
            nondot_names |= self.KLRW_edge_param_names(
                prefixes.get("edge_prefix", "t"), prefixes.get("framing_prefix", "u")
            )
        dot_names = {}
        dot_names |= self.KLRW_dots_names(prefixes.get("dots_prefix", "x"))
        if not no_deformations:
            dot_names |= self.KLRW_deformations_names(
                prefixes.get("deformations_prefix", "z")
            )

        return nondot_names, dot_names

    def immutable_copy(self):
        return FramedDynkinDiagram_with_dimensions_immutable(
            quiver=self.quiver,
            dimensions_dict=self.dimensions_dict,
        )


class FramedDynkinDiagram_with_dimensions_immutable(
    FramedDynkinDiagram_with_dimensions
):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        # bypass protection in frozen=True to change the entry
        super(FramedDynkinDiagram_with_dimensions, self).__setattr__(
            "dimensions_dict",
            MappingProxyType(self.dimensions_dict.copy()),
        )

    def __hash__(self):
        """
        We need hash to be able pass instances of this class to __init__'s
        of UniqueRepresentation classes [e.g. KLRW algebra]
        """
        return hash((self.dimensions_list(), self.quiver.cartan_type()))

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.quiver.cartan_type(),
                self.quiver,
                dict(self.dimensions_dict),
            ),
        )


class QuiverGradingLabel:
    pass


@dataclass(frozen=True, repr=False)
class QuiverGradingEquvariantLabel(QuiverGradingLabel):
    name: str

    def __repr__(self):
        return repr(self.name)


@dataclass(frozen=True, repr=False)
class QuiverGradingSelfCrossingLabel(QuiverGradingLabel):
    vertex: NodeInFramedQuiver

    def __repr__(self):
        return repr(self.vertex) + ",c"


@dataclass(frozen=True, repr=False)
class QuiverGradingUnorientedCrossingLabel(QuiverGradingLabel):
    edge: NonOrientedEdge | None = None
    vertex_tails: InitVar[NodeInFramedQuiver | None] = None
    vertex_heads: InitVar[NodeInFramedQuiver | None] = None

    def __post_init__(
        self,
        vertex_tails: NodeInFramedQuiver | None,
        vertex_heads: NodeInFramedQuiver | None,
    ):
        assert (vertex_tails is None) == (
            vertex_heads is None
        ), "Only one vertex is assigned"

        # if vertices were given, we need to make sure the edge is right
        if vertex_tails is not None:
            if self.edge is not None:
                assert NonOrientedEdge(vertex_tails, vertex_heads) == self.edge
            else:
                # bypass protection in frozen=True to change the entry
                super(QuiverGradingUnorientedCrossingLabel, self).__setattr__(
                    "edge",
                    NonOrientedEdge(vertex_tails, vertex_heads),
                )

    def __repr__(self):
        return repr(self.edge)


@dataclass(frozen=True, repr=False)
class QuiverGradingDotLabel(QuiverGradingLabel):
    vertex: NodeInFramedQuiver

    def __repr__(self):
        return repr(self.vertex) + ",d"


class QuiverGradingGroupElement(IndexedFreeModuleElement):
    def ordinary_grading(self, as_scalar=True):
        parent = self.parent()
        equivariant_grading_label = parent.equivariant_grading_label
        if as_scalar:
            return self.coefficient(equivariant_grading_label)
        else:
            basis_vector = parent.monomial(equivariant_grading_label)
            return self.ordinary_grading(as_scalar=True) * basis_vector

    def extra_gradings(self):
        return self - self.ordinary_grading(as_scalar=False)


class QuiverGradingGroup(CombinatorialFreeModule):
    def __init__(
        self,
        quiver: FramedDynkinDiagram_class,
        R: Ring = ZZ,
        equivariant_grading_name=0,
        vertex_scaling=True,
        edge_scaling=True,
        dot_scaling=True,
        **kwrds,
    ):
        self.equivariant_grading_label = QuiverGradingEquvariantLabel(
            equivariant_grading_name
        )
        self.vertex_scaling = vertex_scaling
        self.edge_scaling = edge_scaling
        self.dot_scaling = dot_scaling
        self.quiver = quiver
        names = []
        if vertex_scaling:
            names += [
                QuiverGradingSelfCrossingLabel(v) for v in quiver.non_framing_nodes()
            ]
        if edge_scaling:
            # we keep only non-oriented pairs and remove doubles
            edges = set(NonOrientedEdge(v1, v2) for v1, v2, _ in quiver.edges())
            names += [QuiverGradingUnorientedCrossingLabel(e) for e in edges]
        if dot_scaling:
            names += [QuiverGradingDotLabel(v) for v in quiver.non_framing_nodes()]
        names += [self.equivariant_grading_label]

        if "prefix" not in kwrds:
            kwrds["prefix"] = "d_"
        if "bracket" not in kwrds:
            kwrds["bracket"] = "{"

        super().__init__(R, names, element_class=QuiverGradingGroupElement, **kwrds)

    def crossing_grading(self, vertex1, vertex2):
        result = self.term(
            self.equivariant_grading_label,
            -self.quiver.scalar_product_of_simple_roots(vertex1, vertex2),
        )
        if vertex1 == vertex2:
            vertex_label = QuiverGradingSelfCrossingLabel(vertex1)
            if vertex_label in self.indices():
                result += self.monomial(vertex_label)
        else:
            edge = NonOrientedEdge(vertex1, vertex2)
            edge_label = QuiverGradingUnorientedCrossingLabel(edge)
            if edge_label in self.indices():
                result += self.monomial(edge_label)
        return result

    def dot_algebra_grading(self, index):
        result = self.zero()

        if isinstance(index, DotVariableIndex):
            vertex = index.vertex
            if not vertex.is_framing():
                result += self.term(
                    self.equivariant_grading_label,
                    self.quiver.scalar_product_of_simple_roots(vertex, vertex),
                )
                dot_grading_index = QuiverGradingDotLabel(vertex)
                if dot_grading_index in self.indices():
                    result += self.monomial(dot_grading_index)

        elif isinstance(index, VertexVariableIndex):
            vertex = index.vertex
            vertex_grading_index = QuiverGradingSelfCrossingLabel(vertex)
            if vertex_grading_index in self.indices():
                result += self.monomial(vertex_grading_index)
            dot_grading_index = QuiverGradingDotLabel(vertex)
            if dot_grading_index in self.indices():
                result += self.monomial(dot_grading_index)

        elif isinstance(index, EdgeVariableIndex):
            edge = NonOrientedEdge(index.vertex_tail, index.vertex_head)
            edge_grading_index = QuiverGradingUnorientedCrossingLabel(edge)
            result = self.zero()
            if edge_grading_index in self.indices():
                result += 2 * self.monomial(edge_grading_index)
            if not index.vertex_tail.is_framing():
                dot_grading_index = QuiverGradingDotLabel(index.vertex_tail)
                if dot_grading_index in self.indices():
                    result += self.term(
                        dot_grading_index,
                        self.quiver[index.vertex_tail, index.vertex_head],
                    )

        else:
            raise ValueError("Unknown variable type: {}".format(index.__class__))

        return result

    def _coerce_map_from_(self, other):
        """
        Integers are treated as ordinary equivariant gradings
        """
        if other == ZZ:
            return lambda parent, x: self.term(self.equivariant_grading_label, x)

        if isinstance(other, QuiverGradingGroup):
            if self.quiver == other.quiver:
                can_coerce = True
                if self.vertex_scaling is True:
                    can_coerce &= self.vertex_scaling == other.vertex_scaling
                if self.edge_scaling is True:
                    can_coerce &= self.edge_scaling == other.edge_scaling
                if can_coerce:
                    map = {}
                    for index in other.indices():
                        if index == other.equivariant_grading_label:
                            map[index] = self.monomial(self.equivariant_grading_label)
                        elif index in self.indices():
                            map[index] = self.monomial(index)
                        else:
                            map[index] = self.zero()

                    def on_basis(index, map_dict=map):
                        return map_dict[index]

                    return other.module_morphism(on_basis=on_basis, codomain=self)


@dataclass(slots=True)
class QuiverParameter:
    # index : object
    name: str
    position: int
    monomial: object
    invertible: bool = True

    def __repr__(self):
        return self.monomial.__repr__()


class VariableIndex:
    pass


@dataclass(frozen=True)
class DotVariableIndex(VariableIndex):
    vertex: NodeInFramedQuiver
    number: int | None

    def __repr__(self):
        return "Dot index {} #{}".format(self.vertex, self.number)


@dataclass(frozen=True)
class VertexVariableIndex(VariableIndex):
    vertex: NodeInFramedQuiver

    def __repr__(self):
        return "Vertex index {}".format(self.vertex)


@dataclass(frozen=True)
class EdgeVariableIndex(VariableIndex):
    vertex_tail: NodeInFramedQuiver
    vertex_head: NodeInFramedQuiver

    def __repr__(self):
        return "Edge index {}->{}".format(self.vertex_tail, self.vertex_head)


def exps_for_dots_of_degree(etuples_degrees: MappingProxyType | tuple, degree):
    if isinstance(etuples_degrees, MappingProxyType):
        etuples_degrees = tuple(x for x in etuples_degrees.items())
    et, deg = etuples_degrees[0]
    if len(etuples_degrees) == 1:
        if int(degree) % int(deg) == 0:
            yield et.emul(int(degree) // int(deg))
        return
    else:
        power_of_first = int(degree) // int(deg)
        for power in range(power_of_first + 1):
            for rec_et in exps_for_dots_of_degree(
                etuples_degrees[1:],
                degree - power * deg,
            ):
                yield rec_et.eadd_scaled(et, power)


# TODO: add an abstract class KLRWDotsAlgebra(CoomutativeRing, UniqueRep),
# inherit upstairs and downstairs versions
class KLRWDotsAlgebra(CommutativeRing, UniqueRepresentation):
    @staticmethod
    def __classcall_private__(cls, *args, invertible_parameters=False, **kwargs):
        if not invertible_parameters:
            # TODO: add the case when the list of invertible parameters is empty
            return KLRWUpstairsDotsAlgebra(
                invertible_parameters=invertible_parameters, *args, **kwargs
            )
        else:
            return KLRWUpstairsDotsAlgebra_invertible_parameters(
                invertible_parameters=invertible_parameters, *args, **kwargs
            )

    def __init__(
        self,
        base_ring,
        quiver_data: FramedDynkinDiagram_with_dimensions,
        no_deformations=True,
        default_vertex_parameter=None,
        default_edge_parameter=None,
        invertible_parameters=False,
        order="degrevlex",
        **prefixes,
    ):
        # remember initialization data for pickle/unpickle
        self.quiver_data = quiver_data.immutable_copy()
        self.no_deformations = no_deformations
        self.default_vertex_parameter = default_vertex_parameter
        self.default_edge_parameter = default_edge_parameter
        self.invertible_parameters = invertible_parameters
        self.prefixes = prefixes

        no_vertex_parameters = default_vertex_parameter is not None
        no_edge_parameters = default_edge_parameter is not None
        nondot_names, dot_names = quiver_data.names(
            no_deformations=no_deformations,
            no_vertex_parameters=no_vertex_parameters,
            no_edge_parameters=no_edge_parameters,
            **prefixes,
        )

        nondot_names_list = [name for _, name in nondot_names.items()]
        dot_names_list = [name for _, name in dot_names.items()]
        self.init_algebra(base_ring, nondot_names_list, dot_names_list, order)

        # type of variables is uniquely determined by its index data
        self.variables = {
            index: QuiverParameter(
                name,
                position,
                self(name),
                invertible_parameters,
            )
            for position, (index, name) in enumerate(nondot_names.items())
        }
        self.variables |= {
            index: QuiverParameter(
                name,
                position + len(self.variables),  # dots follow non-dots
                self(name),
                False,  # dots are not invertible
            )
            for position, (index, name) in enumerate(dot_names.items())
        }

        # adds default values to self.variables
        self.assign_default_values(
            quiver_data,
            default_vertex_parameter=default_vertex_parameter,
            default_edge_parameter=default_edge_parameter,
            no_deformations=no_deformations,
        )

        # make immutable
        self.variables = MappingProxyType(self.variables)

    def __reduce__(self):
        from functools import partial

        return (
            partial(self.__class__, **self.prefixes),
            (
                self.base(),
                self.quiver_data,
                self.no_deformations,
                self.default_vertex_parameter,
                self.default_edge_parameter,
                self.invertible_parameters,
                self.term_order().name(),
            ),
        )

    def assign_default_values(
        self,
        quiver_data,
        no_deformations=True,
        default_vertex_parameter=None,
        default_edge_parameter=None,
    ):
        # TODO: add default parameters' options,
        # maybe by changing self.variables to a custom dictionary
        # second set of framing variables rescaled to 1
        quiver = quiver_data.quiver
        if default_edge_parameter is None:
            for n1, n2 in product(quiver.vertices(), quiver.vertices()):
                index = EdgeVariableIndex(n1, n2)
                if index not in self.variables:
                    self.variables[index] = QuiverParameter(
                        None, None, self.one(), True
                    )
        else:
            default_value = self(default_edge_parameter)
            is_unit = default_value.is_unit()
            for n1, n2 in product(quiver.vertices(), quiver.vertices()):
                index = EdgeVariableIndex(n1, n2)
                self.variables[index] = QuiverParameter(
                    None, None, default_value, is_unit
                )
        if default_vertex_parameter is not None:
            default_value = self(default_vertex_parameter)
            is_unit = default_value.is_unit()
            for key in quiver.non_framing_nodes():
                index = VertexVariableIndex(key)
                self.variables[index] = QuiverParameter(
                    None, None, default_value, is_unit
                )
        if no_deformations:
            for key in quiver_data.KLRW_deformations_names():
                self.variables[key] = QuiverParameter(None, None, self.zero(), False)

    def center_gens(self):
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    continue
            if variable.name is not None:
                yield variable.monomial
        yield from self.symmetric_dots_gens()

    def basis_modulo_symmetric_dots(self):
        variables = []
        degree_ranges = []
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    degree_ranges.append(range(index.number))
                    variables.append(variable.monomial)
        degree_iterator = product(*degree_ranges)
        for degree in degree_iterator:
            yield prod(var**deg for var, deg in zip(variables, degree))

    @lazy_attribute
    def number_of_non_dot_params(self):
        return sum(1 for index in self.names if not isinstance(index, DotVariableIndex))

    @cached_method
    def dot_exptuples_and_degrees(self, grading_group: QuiverGradingGroup):
        result = {}
        relevant_grading_index = grading_group.equivariant_grading_label
        for index, var in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    exps = var.monomial.exponents()
                    # must be a monomial
                    assert len(exps) == 1, repr(var)
                    grading = grading_group.dot_algebra_grading(index)
                    result[exps[0]] = grading.coefficient(relevant_grading_index)
        return MappingProxyType(result)

    @cached_method
    def degrees_by_position(self, grading_group: QuiverGradingGroup):
        result = [list() for _ in range(self.ngens())]
        for index, var in self.variables.items():
            if var.position is not None:
                result[var.position] = grading_group.dot_algebra_grading(index)

        return tuple(result)

    @lazy_attribute
    def dot_variables(self):
        result = defaultdict(list)
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                result[index[0]].append(variable)

        # make everything immutable
        for key in result:
            result[key] = tuple(result[key])

        return MappingProxyType(result)

    def symmetric_dots_gens(self):
        elementary_symmetric_polynomials = {}
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    if index.vertex in elementary_symmetric_polynomials:
                        e_list = elementary_symmetric_polynomials[index.vertex]
                        # add the top elementary symmetric polynomial
                        e_list.append(variable.monomial * e_list[-1])
                        # modify intermediate symmetric polynomials
                        for i in range(len(e_list) - 2, 0, -1):
                            e_list[i] += variable.monomial * e_list[i - 1]
                        # modify the first elementary symmetric polynomial
                        e_list[0] += variable.monomial
                    else:
                        # for one variable there is only
                        # the first elementary symmetric polynomial
                        elementary_symmetric_polynomials[index.vertex] = [
                            variable.monomial
                        ]

        for _, e_list in elementary_symmetric_polynomials.items():
            yield from e_list

    def basis_by_degree(self, degree):
        return tuple(self.monomial(*et) for et in self.exps_by_degree(degree))

    @cached_method
    def exps_by_degree(self, degree):
        """
        Return a tuple of exponents of a given degree.

        Monomials of these degrees form a basis over
        the subalgebra of elements of zero degree.
        """

        # we make a polynomial with monomials representing all
        # possible monomials of given degree
        result = self.one()
        # first we take into account "easy" degrees
        # where powers of varibles are uniquely reconstructed
        grading_group = degree.parent()
        degrees_taken_into_account = grading_group.zero()
        for label, coeff in degree:
            if isinstance(label, QuiverGradingSelfCrossingLabel):
                index = VertexVariableIndex(label.vertex)
                variable = self.variables[index]
                if coeff < 0:
                    if not variable.invertible:
                        # we prohibit inverse powers
                        return tuple()
                degrees_taken_into_account += coeff * grading_group.dot_algebra_grading(
                    index
                )
                result *= variable.monomial**coeff

            elif isinstance(label, QuiverGradingUnorientedCrossingLabel):
                if int(coeff) % int(2) != 0:
                    return tuple()
                d = int(coeff) // int(2)
                v1, v2 = label.edge
                edge_index = EdgeVariableIndex(v1, v2)
                edge_opp_index = EdgeVariableIndex(v2, v1)
                if v2.is_framing():
                    variable = self.variables[edge_index]
                    degree_part = grading_group.dot_algebra_grading(edge_index)
                elif v1.is_framing():
                    variable = self.variables[edge_opp_index]
                    degree_part = grading_group.dot_algebra_grading(edge_opp_index)
                elif not grading_group.dot_scaling:
                    # in this case t_ij and t_ji have the same grading,
                    # t_ij/t_ji is in the subalgebra of degree zero elements
                    variable = self.variables[edge_index]
                    degree_part = grading_group.dot_algebra_grading(edge_index)
                else:
                    # in other cases this degree has to be treated later by an ILP.
                    continue

                if d < 0:
                    if not variable.invertible:
                        return tuple()
                result *= variable.monomial**d
                degrees_taken_into_account += d * degree_part

            elif label != grading_group.equivariant_grading_label and not isinstance(
                label, QuiverGradingDotLabel
            ):
                raise ValueError("Unknown type of grading: {}".format(label))

        # take care of dots if they are only constrained by ordinary equivariant degree
        if not grading_group.dot_scaling:
            excess_degree = degree - degrees_taken_into_account
            coeff = excess_degree.coefficient(grading_group.equivariant_grading_label)
            if int(coeff) % int(2) != 0 or coeff < 0:
                return tuple()
            new_part = sum(
                self.monomial(*etuple)
                for etuple in exps_for_dots_of_degree(
                    self.dot_exptuples_and_degrees(grading_group),
                    coeff,
                )
            )
            degrees_taken_into_account += grading_group.term(
                grading_group.equivariant_grading_label,
                coeff,
            )
            result *= new_part

        excess_degree = degree - degrees_taken_into_account
        if not excess_degree.is_zero():
            excess_part = self.zero()
            indices_inv, indices_non_inv = self._indices_for_ilp_on_gradings_()
            indices = indices_inv + indices_non_inv
            solutions = self._solve_ilp_on_gradings_(excess_degree)
            if solutions.shape[0] == 0:
                return tuple()
            for i in range(solutions.shape[0]):
                part_from_a_solution = self.one()
                for j, index in enumerate(indices):
                    if isinstance(index, EdgeVariableIndex):
                        part_from_a_solution *= (
                            self.variables[index].monomial ** solutions[i, j]
                        )
                    elif isinstance(index, DotVariableIndex):
                        part_from_a_solution *= (
                            self.complete_homogeneous_symmetric_in_dots(
                                vertex_of_dots=index.vertex, n=solutions[i, j]
                            )
                        )
                    else:
                        raise ValueError("Unxepected variable index: {}".format(index))
                excess_part += part_from_a_solution
            result *= excess_part

        return tuple(x for x in result.exponents())

    @cached_method
    def _indices_for_ilp_on_gradings_(self):
        relevant_invertible_indices = set()
        relevant_not_invertible_indices = set()
        for index, var in self.variables.items():
            if var.name is not None:
                if isinstance(index, DotVariableIndex):
                    dot_index = replace(index, number=1)
                    relevant_not_invertible_indices.add(dot_index)
                if isinstance(index, EdgeVariableIndex):
                    if (
                        not index.vertex_head.is_framing()
                        and not index.vertex_tail.is_framing()
                    ):
                        if var.invertible:
                            relevant_invertible_indices.add(index)
                        else:
                            relevant_not_invertible_indices.add(index)

        relevant_invertible_indices = tuple(relevant_invertible_indices)
        relevant_not_invertible_indices = tuple(relevant_not_invertible_indices)

        return relevant_invertible_indices, relevant_not_invertible_indices

    @cached_method
    def _ilp_on_gradings_(self, grading_group):
        from scipy.sparse import dok_array
        from numpy import intc

        relevant_invertible_indices, relevant_not_invertible_indices = (
            self._indices_for_ilp_on_gradings_()
        )

        A_inv = dok_array(
            (
                grading_group.dimension(),
                len(relevant_invertible_indices),
            ),
            dtype=intc,
        )
        for j in range(len(relevant_invertible_indices)):
            index = relevant_invertible_indices[j]
            monomial = self.variables[index].monomial
            degree = self.element_degree(monomial, grading_group)
            for i, v in degree._vector_(sparse=True).dict(copy=False).items():
                A_inv[i, j] = v

        A_non_inv = dok_array(
            (
                grading_group.dimension(),
                len(relevant_not_invertible_indices),
            ),
            dtype=intc,
        )
        for j in range(len(relevant_not_invertible_indices)):
            index = relevant_not_invertible_indices[j]
            monomial = self.variables[index].monomial
            degree = self.element_degree(monomial, grading_group)
            for i, v in degree._vector_(sparse=True).dict(copy=False).items():
                A_non_inv[i, j] = v

        return A_inv.tocsr(), A_non_inv.tocsr()

    @cached_method
    def _solve_ilp_on_gradings_(self, degree):
        from scipy.sparse import dok_array
        from numpy import intc, zeros
        import gurobipy as gp

        grading_group = degree.parent()

        A_inv, A_non_inv = self._ilp_on_gradings_(grading_group)

        y_transposed = dok_array((1, grading_group.dimension()), dtype=intc)
        for i, v in degree._vector_(sparse=True).dict(copy=False).items():
            y_transposed[0, i] = v

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as m:
                m.Params.PoolSearchMode = 2
                m.Params.PoolSolutions = gp.GRB.MAXINT

                x_non_inv = m.addMVar(A_non_inv.shape[1], lb=int(0), vtype="I")
                x_inv = m.addMVar(
                    A_inv.shape[1],
                    lb=-float("inf"),
                    vtype="I",
                )

                if A_inv.shape[1] == 0:
                    m.addConstr(A_non_inv @ x_non_inv == y_transposed)
                elif A_non_inv.shape[1] == 0:
                    m.addConstr(A_inv @ x_inv == y_transposed)
                else:
                    m.addConstr(A_inv @ x_inv + A_non_inv @ x_non_inv == y_transposed)

                m.optimize()
                # if the system is inconsistent
                if m.Status == 3:
                    return zeros(
                        shape=(0, x_inv.size + x_non_inv.size),
                        dtype=intc,
                    )

                n_solutions = m.SolCount

                solutions = zeros(
                    shape=(n_solutions, x_inv.size + x_non_inv.size),
                    dtype=intc,
                )
                for n in range(n_solutions):
                    m.Params.SolutionNumber = n
                    solutions[n, : x_inv.size] = x_inv.Xn
                    solutions[n, x_inv.size :] = x_non_inv.Xn

        return solutions

    def exp_degree(self, exp: ETuple, grading_group: QuiverGradingGroup):
        degrees_by_position = self.degrees_by_position(grading_group)
        return grading_group.linear_combination(
            (degrees_by_position[position], power)
            for position, power in exp.sparse_iter()
        )

    # libsingular wrapper in Sage does not allow to change elements for a wrapper,
    # so we define degree in the parent
    def element_degree(
        self, element, grading_group: QuiverGradingGroup, check_if_homogeneous=False
    ):
        degrees_by_position = self.degrees_by_position(grading_group)
        # zero elements return None as degree
        degree = None
        for exp, scalar in element.iterator_exp_coeff():
            if not scalar.is_zero():
                term_degree = grading_group.linear_combination(
                    (degrees_by_position[position], power)
                    for position, power in exp.sparse_iter()
                )

                if not check_if_homogeneous:
                    return term_degree
                elif degree is None:
                    degree = term_degree
                elif degree != term_degree:
                    raise ValueError("The dots are not homogeneous! {}".format(element))

        return degree

    def is_element_symmetric(self, element):
        for v in self.quiver_data.non_framing_nodes():
            dim_v = self.quiver_data[v]
            for i in range(1, dim_v):
                x_i = self.dot_variable(v, i)
                x_i_1 = self.dot_variable(v, i + 1)
                swap = {
                    x_i.name: x_i_1.monomial,
                    x_i_1.name: x_i.monomial,
                }
                swapped_element = element.subs(**swap)
                if swapped_element != element:
                    return False
        return True

    def etuple_ignoring_dots(self, et: ETuple):
        # we assume that the non-zero dots have first positions
        return et[: self.number_of_non_dot_params]

    def hom_from_dots_map(self, codomain, map: MappingProxyType):
        variables_images = [
            (
                codomain.variables[map[index]].monomial
                if index in map
                else codomain.variables[index].monomial
            )
            for index in self.variables
            if self.variables[index].name is not None
        ]

        return self.hom(variables_images, codomain)

    def _coerce_map_from_(self, other):
        if isinstance(other, KLRWDotsAlgebra):
            if self.quiver_data == other.quiver_data:
                can_coerce = self.base().has_coerce_map_from(other.base())
                if other.default_vertex_parameter is not None:
                    can_coerce &= (
                        self.default_vertex_parameter == other.default_vertex_parameter
                    )
                if other.default_edge_parameter is not None:
                    can_coerce &= (
                        self.default_edge_parameter == other.default_edge_parameter
                    )
                if can_coerce:
                    variables_images = [None] * other.ngens()
                    for index, var in other.variables.items():
                        if var.name is not None:
                            var_image = self.variables[index].monomial
                            variables_images[var.position] = var_image
                    return other.hom(tuple(variables_images), self)
            return

        return super()._coerce_map_from_(other)

    @cached_method
    def hom_in_simple(self):
        """
        Kill dots, set all multiplicative parameters to 1.

        Gets scalars in a ring R.
        """
        variables_images = []
        for index, var in self.variables.items():
            if var.name is not None:
                if isinstance(index, DotVariableIndex):
                    # setting all dots to zero
                    variables_images.append(self.base().zero())
                else:
                    # setting all other parameters to 1
                    variables_images.append(self.base().one())

        return self.hom(variables_images, self.base())

    def variable(self, *index):
        if len(index) == 1:
            assert isinstance(
                index[0], NodeInFramedQuiver
            ), "Unknown index type, {}".format(index)
            return self.vertex_variable(index[0])

        assert len(index) == 2, "Unknown index type, {}".format(index)

        if isinstance(index[1], NodeInFramedQuiver):
            return self.edge_variable(*index)
        else:
            return self.dot_variable(*index)

    def dot_variable(self, vertex, number):
        return self.variables[DotVariableIndex(vertex=vertex, number=number)]

    def vertex_variable(self, vertex):
        return self.variables[VertexVariableIndex(vertex=vertex)]

    def edge_variable(self, vertex_tail, vertex_head):
        return self.variables[
            EdgeVariableIndex(vertex_tail=vertex_tail, vertex_head=vertex_head)
        ]

    @cached_method
    def complete_homogeneous_symmetric_in_dots(
        self, vertex_of_dots: NodeInFramedQuiver, n: int
    ):
        from sage.combinat.sf.sf import SymmetricFunctions

        h_n = SymmetricFunctions(self.base()).complete()([n])
        n_vars = self.quiver_data[vertex_of_dots]
        h_n = h_n.expand(n=n_vars)
        images = [self.dot_variable(vertex_of_dots, i + 1) for i in range(n_vars)]
        m = h_n.parent().hom(images, self)
        return m(h_n)


class KLRWUpstairsDotsAlgebra(KLRWDotsAlgebra, PolynomialRing_sing):
    def init_algebra(self, base_ring, nondot_names_list, dot_names_list, order):
        super(KLRWDotsAlgebra, self).__init__(
            base_ring,
            len(nondot_names_list) + len(dot_names_list),
            nondot_names_list + dot_names_list,
            order,
        )


#class DotAlgebraElement_invertible_parameters(LaurentPolynomial_mpair):
    # TODO: can speed up __init__ by using reduce=False
    # need Cython
#    pass
    # def __getattribute__(self, name):
    #     print("Element :: Accessing attribute {}".format(name))
    #     return super().__getattribute__(name)


class KLRWUpstairsDotsAlgebra_invertible_parameters(
    KLRWDotsAlgebra, LaurentPolynomialRing
):
#    Element = DotAlgebraElement_invertible_parameters

    def init_algebra(self, base_ring, nondot_names_list, dot_names_list, order):
        polynomial_ring = PolynomialRing(
            base_ring,
            len(nondot_names_list) + len(dot_names_list),
            nondot_names_list + dot_names_list,
            order=order,
        )
        super(KLRWDotsAlgebra, self).__init__(polynomial_ring)

    # def __getattribute__(self, name):
    #    print("Algebra :: Accessing attribute {}".format(name))
    #    return super().__getattribute__(name)
