from typing import NamedTuple
from collections import defaultdict
from dataclasses import dataclass, InitVar
from types import MappingProxyType

from sage.combinat.root_system.dynkin_diagram import DynkinDiagram_class
from sage.combinat.root_system.cartan_type import CartanType_abstract
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.sage_object import SageObject

from sage.misc.cachefunc import cached_method


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

    @cached_method
    def framing_nodes(self):
        return tuple(v.make_framing() for v in self.non_framing_nodes())

    def inject_nodes(self, scope=None, verbose=True):
        """
        Defines globally nodes
        """
        vs = [v.__repr__() for v in self.vertices()]
        gs = [v for v in self.vertices()]
        if scope is None:
            import inspect
            # Sage in inject_variables relies of old style globals convention in Cython.
            # We replace it with the help of https://stackoverflow.com/a/40690954
            callerglobals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
            scope = callerglobals
        if verbose:
            print("Defining %s" % (", ".join(vs)))
        for v, g in zip(vs, gs):
            scope[v] = g

    def KLRW_vertex_param_names(self, vertex_prefix="r"):
        from klrw.dot_algebra import VertexVariableIndex
        vertex_names = {}
        # iterating over non-framing vertices
        for v in self.non_framing_nodes():
            vertex_names[VertexVariableIndex(v)] = vertex_prefix + "_{}".format(v.node)

        return vertex_names

    def KLRW_edge_param_names(self, edge_prefix="t", framing_prefix="u"):
        from klrw.dot_algebra import EdgeVariableIndex
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

    @classmethod
    def with_zero_dimensions(cls, quiver):
        if isinstance(quiver, FramedDynkinDiagram_class):
            return cls(quiver=quiver)
        elif isinstance(quiver, CartanType_abstract):
            return cls(ct=quiver)
        else:
            raise ValueError(
                "Parameter `quiver` must be a Cartan Type "
                + "or Dynkin Diagram, not {}".format(quiver.__class__)
            )

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

    def framing_nodes(self, *args, **kwargs):
        return self.quiver.framing_nodes(*args, **kwargs)

    def inject_nodes(self, scope=None, verbose=True):
        if scope is None:
            import inspect
            # Sage in inject_variables relies of old style globals convention in Cython.
            # We replace it with the help of https://stackoverflow.com/a/40690954
            callerglobals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
            scope = callerglobals
        self.quiver.inject_nodes(scope=scope, verbose=verbose)

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
        from klrw.dot_algebra import DotVariableIndex

        dots_names = {}
        # iterating over non-framing vertices
        for v in self.quiver.non_framing_nodes():
            for k in range(self.dimensions_dict[v]):
                dots_names[DotVariableIndex(v, k + 1)] = dots_prefix + "_{}_{}".format(
                    v.node, k + 1
                )
        return dots_names

    def KLRW_deformations_names(self, deformations_prefix="z"):
        from klrw.dot_algebra import DotVariableIndex

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

    def mutable_copy(self):
        return FramedDynkinDiagram_with_dimensions(
            ct=self.quiver.cartan_type(),
            quiver=self.quiver,
            dimensions_dict=dict(self.dimensions_dict),
        )
