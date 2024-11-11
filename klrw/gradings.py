from dataclasses import dataclass, InitVar  # , replace

from sage.rings.ring import Ring
from sage.rings.integer_ring import ZZ
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement

from klrw.framed_dynkin import (
    NodeInFramedQuiver,
    NonOrientedEdge,
    FramedDynkinDiagram_class,
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
        from klrw.dot_algebra import (
            DotVariableIndex,
            VertexVariableIndex,
            EdgeVariableIndex,
        )

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

    def _element_constructor_(self, *args):
        if len(args) == 1:
            if isinstance(args[0], int):
                x = ZZ(args[0])
                return self.coerce(x)
        super()._element_constructor_(*args)
