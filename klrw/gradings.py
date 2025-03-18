from typing import Iterable, Any
from dataclasses import dataclass, InitVar
from functools import total_ordering

from sage.rings.ring import Ring
from sage.rings.integer_ring import ZZ
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.structure.element import get_coercion_model

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


@total_ordering
@dataclass(frozen=True, repr=False)
class QuiverGradingHomologicalLabel(QuiverGradingLabel):
    name: Any

    def __repr__(self):
        if self.name is not None:
            return "h_" + repr(self.name)
        return "h"

    def __lt__(self, other):
        return self.name < other.name


class HomologicalGradingGroupElement(IndexedFreeModuleElement):
    pass


@dataclass(frozen=True, eq=True)
class SignMorphism:
    contributing_generators: frozenset[QuiverGradingLabel]

    def __post_init__(self):
        if not isinstance(self.contributing_generators, frozenset):
            super().__setattr__(
                "contributing_generators",
                frozenset(self.contributing_generators),
            )

    def __call__(self, grading):
        from sage.rings.finite_rings.integer_mod_ring import Zmod

        coeff = sum(
            Zmod(2)(grading.coefficient(label))
            for label in self.contributing_generators
        )
        if coeff:
            return ZZ(-1)
        else:
            return ZZ(1)


class HomologicalGradingGroup(CombinatorialFreeModule):
    @staticmethod
    def __classcall_private__(
        cls,
        R: Ring = ZZ,
        homological_grading_names: Iterable[Any] = (None,),
        **kwrds,
    ):
        homological_grading_names = tuple(homological_grading_names)
        return super().__classcall__(
            cls,
            R=R,
            homological_grading_names=homological_grading_names,
            **kwrds,
        )

    def __init__(
        self,
        R: Ring,
        homological_grading_names: tuple,
        **kwrds,
    ):
        names = [
            QuiverGradingHomologicalLabel(name) for name in homological_grading_names
        ]
        if len(names) == 1:
            self.homological_grading_label = names[0]

        if "prefix" not in kwrds:
            kwrds["prefix"] = "d_"
        if "bracket" not in kwrds:
            kwrds["bracket"] = "{"

        super().__init__(
            R, names, element_class=HomologicalGradingGroupElement, **kwrds
        )

    def names(self):
        return tuple(index.name for index in self.indices())

    def _element_constructor_(self, *args):
        if len(args) == 1:
            if isinstance(args[0], int):
                x = ZZ(args[0])
                return self.coerce(x)
        super()._element_constructor_(*args)

    def _coerce_map_from_(self, other):
        """
        Integers are treated as ordinary homological gradings
        """
        if other == ZZ:
            return lambda parent, x: self.term(self.homological_grading_label, x)

        if isinstance(other, HomologicalGradingGroup):
            if all(label in self.indices() for label in other.indices()):
                map = {}
                for index in other.indices():
                    if index in self.indices():
                        map[index] = self.monomial(index)
                    else:
                        map[index] = self.zero()

                def on_basis(index, map_dict=map):
                    return map_dict[index]

                return other.module_morphism(on_basis=on_basis, codomain=self)

    def _convert_map_from_(self, other):
        """
        Integers are treated as ordinary homological gradings
        """
        if isinstance(other, ExtendedQuiverGradingGroup):
            if other.homological_part == self:
                return other.to_homological_part

    @lazy_attribute
    def default_sign_morphism(self):
        try:
            homological_grading_label = self.homological_grading_label
        except AttributeError:
            raise NotImplementedError("Need a different implementation of sign.")

        return SignMorphism([homological_grading_label])

    @staticmethod
    def direct_sum(*summands):
        return DirectSumOfHomologicalGradingGroup(*summands)

    def __add__(self, other):
        return self.direct_sum(self, other)

    @lazy_attribute
    def totalization_morphism(self):
        """
        All homological gradings are treated as one.
        """
        # only one grading
        codomain = HomologicalGradingGroup(R=self.base())
        gen = codomain.gens()[0]

        return self._module_morphism(
            on_basis=lambda _: gen,
            codomain=codomain,
        )


class ExtendedQuiverGradingGroupElement(IndexedFreeModuleElement):
    def homological_part(self):
        return self.parent().to_homological_part(self)

    def equivariant_part(self):
        return self.parent().to_equivariant_part(self)

    def ordinary_grading(self, as_scalar=True):
        """
        Return ordinary equivariant grading.
        """
        return self.equivariant_part().ordinary_grading(as_scalar)

    def extra_gradings(self):
        return self.equivariant_part().extra_gradings()


class ExtendedQuiverGradingGroup(CombinatorialFreeModule):
    @staticmethod
    def __classcall_private__(
        cls,
        equivariant_grading_group: QuiverGradingGroup,
        homological_grading_group: CombinatorialFreeModule | None = None,
    ):
        if homological_grading_group is None:
            homological_grading_group = HomologicalGradingGroup(R=ZZ)
        return super().__classcall__(
            cls,
            equivariant_grading_group=equivariant_grading_group,
            homological_grading_group=homological_grading_group,
        )

    def __init__(
        self,
        equivariant_grading_group: QuiverGradingGroup,
        homological_grading_group: CombinatorialFreeModule,
    ):
        ring = get_coercion_model().common_parent(
            equivariant_grading_group.base_ring(), homological_grading_group.base_ring()
        )
        # indices have different classes, so they are distinct and we can just join
        names = (
            homological_grading_group.indices().list()
            + equivariant_grading_group.indices().list()
        )
        super().__init__(ring, names, element_class=ExtendedQuiverGradingGroupElement)
        self.homological_part = homological_grading_group
        self.equivariant_part = equivariant_grading_group

    @lazy_attribute
    def to_homological_part(self):
        return self._module_morphism(
            lambda index: (
                self.homological_part.monomial(index)
                if isinstance(index, QuiverGradingHomologicalLabel)
                else self.homological_part.zero()
            ),
            codomain=self.homological_part,
        )

    @lazy_attribute
    def from_homological_part(self):
        return self.homological_part._module_morphism(
            lambda index: self.monomial(index),
            codomain=self,
        )

    @lazy_attribute
    def to_equivariant_part(self):
        return self._module_morphism(
            lambda index: (
                self.equivariant_part.monomial(index)
                if not isinstance(index, QuiverGradingHomologicalLabel)
                else self.equivariant_part.zero()
            ),
            codomain=self.equivariant_part,
        )

    @lazy_attribute
    def from_equivariant_part(self):
        return self.equivariant_part._module_morphism(
            lambda index: self.monomial(index),
            codomain=self,
        )

    def _element_constructor_(self, *args):
        components = None
        if len(args) == 2:
            components = args
            # components = (ZZ(args[0]), ZZ(args[1]))
        if len(args) == 1:
            from collections.abc import Sized

            if isinstance(args[0], int):
                components = (args[0], 0)
            elif isinstance(args[0], Sized) and len(args[0]) == 2:
                components = args[0]
        if components is not None:
            # define the result as the sum of
            # homological + equivariant
            return self.from_homological_part(
                self.homological_part(components[0])
            ) + self.from_equivariant_part(self.equivariant_part(components[1]))

        return super()._element_constructor_(*args)

    def _morphism_from_equivariant_and_homological_part_(
        self,
        domain,
        equivariant_morphism=None,
        homological_morphism=None,
    ):
        """
        Make a morphism from how it acts on equivariant and homological parts.

        If a morphism is not given, it's considered to be the identity.
        """

        def morphism_map(x):
            equivariant_part = x.equivariant_part()
            if equivariant_morphism is not None:
                equivariant_part = equivariant_morphism(equivariant_part)
            equivariant_part = self.from_equivariant_part(equivariant_part)
            homological_part = x.homological_part()
            if homological_morphism is not None:
                homological_part = homological_morphism(homological_part)
            homological_part = self.from_homological_part(homological_part)
            return equivariant_part + homological_part

        return domain.module_morphism(function=morphism_map, codomain=self)

    def _coerce_map_from_(self, other):
        if isinstance(other, ExtendedQuiverGradingGroup):
            if self.equivariant_part.has_coerce_map_from(
                other.equivariant_part
            ) and self.homological_part.has_coerce_map_from(other.homological_part):
                equivariant_part = self.equivariant_part.coerce_map_from(
                    other.equivariant_part
                )
                homological_part = self.homological_part.coerce_map_from(
                    other.homological_part
                )
                return self._morphism_from_equivariant_and_homological_part_(
                    domain=other,
                    equivariant_morphism=equivariant_part,
                    homological_morphism=homological_part,
                )
        return super()._coerce_map_from_(other)

    def _convert_map_from_(self, other):
        """
        Integers are treated as ordinary homological gradings
        """
        if other == self.homological_part:
            return self.from_homological_part
        if other == self.equivariant_part:
            return self.from_equivariant_part

    @lazy_attribute
    def default_sign_morphism(self):
        return self.homological_part.default_sign_morphism

    @staticmethod
    def merge(*summands):
        return MergedExtendedQuiverGradingGroup(*summands)

    def __add__(self, other):
        if isinstance(other, HomologicalGradingGroup):
            other = ExtendedQuiverGradingGroup(
                self.equivariant_part,
                other,
            )
        assert isinstance(other, ExtendedQuiverGradingGroup)
        return self.merge(self, other)

    @lazy_attribute
    def totalization_morphism(self):
        """
        All homological gradings are treated as one.
        """
        hom_totalization_morphism = self.homological_part.totalization_morphism
        codomain = ExtendedQuiverGradingGroup(
            homological_grading_group=hom_totalization_morphism.codomain(),
            equivariant_grading_group=self.equivariant_part,
        )

        return codomain._morphism_from_equivariant_and_homological_part_(
            domain=self,
            homological_morphism=hom_totalization_morphism,
        )


class DirectSumOfHomologicalGradingGroup(HomologicalGradingGroup):
    """
    A direct sum of gradings.
    """

    def __init__(
        self,
        *groups,
    ):
        from klrw.misc import get_from_all_and_assert_equality

        assert all(isinstance(gr, HomologicalGradingGroup) for gr in groups)
        assert groups, "Need at least one grading group"
        R = get_from_all_and_assert_equality(lambda x: x.base_ring(), groups)

        self._parts = groups
        names = tuple(
            self._part_label_to_sum_name_(i, label)
            for i in range(len(groups))
            for label in groups[i].indices()
        )
        HomologicalGradingGroup.__init__(
            self,
            homological_grading_names=names,
            R=R,
        )

    @staticmethod
    def _part_label_to_sum_name_(i: int, label: QuiverGradingHomologicalLabel):
        """
        Make the name of a label from a label for a part.

        If the name of the label is `None`, then the new
        name is the index of the part.
        If the name of the label is not None, the new
        name is a tuple `(i, old_name)`
        """
        if label.name is None:
            return i
        else:
            return (i, label.name)

    @classmethod
    def _part_label_to_sum_label_(cls, i: int, label: QuiverGradingHomologicalLabel):
        return QuiverGradingHomologicalLabel(cls._part_label_to_sum_name_(i, label))

    @staticmethod
    def _sum_label_to_part_label_(label: QuiverGradingHomologicalLabel):
        name = label.name
        if isinstance(name, int):
            return QuiverGradingHomologicalLabel(None)
        else:
            return QuiverGradingHomologicalLabel(name[1])

    @staticmethod
    def _sum_label_to_part_index_(label):
        name = label.name
        if isinstance(name, int):
            return name
        else:
            return name[0]

    @cached_method
    def summand_embedding(self, i):
        """
        Embeds gradings of pieces.
        """
        i = int(i)
        assert i >= 0
        assert i < len(self._parts)
        return self._parts[i]._module_morphism(
            lambda label: self.monomial(self._part_label_to_sum_label_(i, label)),
            codomain=self,
        )

    @cached_method
    def summand_projection(self, i):
        """
        Project to a direct summand.
        """
        i = int(i)
        assert i >= 0
        assert i < len(self._parts)
        module = self._parts[i]
        return self._module_morphism(
            on_basis=lambda sum_label: (
                module.monomial(self._sum_label_to_part_label_(sum_label))
                if i == self._sum_label_to_part_index_(sum_label)
                else module.zero()
            ),
            codomain=module,
        )

    def cartesian_product_of_elements(self, *elements):
        """
        Construct an element from parts.
        """
        return self.sum(
            self.summand_embedding(i)(element) for i, element in enumerate(elements)
        )

    def sign_from_part(self, i, sign_on_part):
        return SignMorphism(
            (
                self._part_label_to_sum_label_(i, label)
                for label in sign_on_part.contributing_generators
            )
        )

    def default_sign_from_part(self, i):
        return self.sign_from_part(i, self._parts[i].default_sign_morphism)


class MergedExtendedQuiverGradingGroup(ExtendedQuiverGradingGroup):
    """
    A direct sum of gradings with identification of the equivariant part.
    """

    def __init__(
        self,
        *groups,
    ):
        from klrw.misc import get_from_all_and_assert_equality

        assert all(isinstance(gr, ExtendedQuiverGradingGroup) for gr in groups)
        assert groups, "Need at least one grading group"
        equivariant_part = get_from_all_and_assert_equality(
            lambda x: x.equivariant_part,
            groups,
        )

        self._parts = groups
        homological_part = DirectSumOfHomologicalGradingGroup(
            *(gr.homological_part for gr in groups)
        )
        ExtendedQuiverGradingGroup.__init__(
            self,
            equivariant_grading_group=equivariant_part,
            homological_grading_group=homological_part,
        )

    @cached_method
    def summand_embedding(self, i):
        """
        Embeds gradings of pieces.
        """
        i = int(i)
        assert i >= 0
        assert i < len(self._parts)
        return self._morphism_from_equivariant_and_homological_part_(
            domain=self._parts[i],
            homological_morphism=self.homological_part.summand_embedding(i),
        )

    @cached_method
    def summand_projection(self, i):
        """
        Project to a direct summand.
        """
        i = int(i)
        assert i >= 0
        assert i < len(self._parts)
        return self._parts[i]._morphism_from_equivariant_and_homological_part_(
            domain=self,
            homological_morphism=self.homological_part.summand_projection(i),
        )

    def cartesian_product_of_elements(self, *elements):
        """
        Construct an element from parts.
        """
        return self.sum(
            self.summand_embedding(i)(element) for i, element in enumerate(elements)
        )
