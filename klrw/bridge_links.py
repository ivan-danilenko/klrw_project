from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from sage.combinat.root_system.root_system import RootSystem
from sage.combinat.root_system.weight_space import WeightSpaceElement
from sage.misc.cachefunc import cached_method


@dataclass(frozen=True)
class StrandEnd:
    strand: int
    is_top: bool

    def __repr__(self):
        result = "top" if self.is_top else "bottom"
        result += " of strand {}".format(self.strand)

        return result


@dataclass(frozen=True)
class BridgeLink:
    bridge_number: int
    braid: Iterable[int]

    def _top_ends_iter_(self):
        for i in range(1, 2 * self.bridge_number + 1):
            yield StrandEnd(strand=i, is_top=True)

    def _bottom_ends_iter_(self):
        for i in range(1, 2 * self.bridge_number + 1):
            yield StrandEnd(strand=i, is_top=False)

    def _ends_iter_(self):
        yield from self._top_ends_iter_()
        yield from self._bottom_ends_iter_()

    def _other_end_of_strand_(self, strand_end: StrandEnd):
        if strand_end.is_top:
            w = reversed(self.braid)
        else:
            w = self.braid
        index = strand_end.strand
        for i in w:
            if index == i:
                index += 1
            elif index == i + 1:
                index -= 1
        return StrandEnd(strand=index, is_top=not strand_end.is_top)

    def _end_connections_iter_(self):
        """
        Edges for _ends_graph_()
        """
        # first list connections by cups and caps
        for is_top in [True, False]:
            for i in range(self.bridge_number):
                yield (
                    StrandEnd(strand=2 * i + 1, is_top=is_top),
                    StrandEnd(strand=2 * i + 2, is_top=is_top),
                )
        # now connections by strands
        for bot_end in self._bottom_ends_iter_():
            yield (bot_end, self._other_end_of_strand_(bot_end))

    @cached_method
    def _ends_graph_(self):
        """
        Makes a graph with ends of strands as objects.

        Two ends are connected if they are ends of the same strand
        or if they are conneced by cups or caps.
        """
        from sage.graphs.graph import Graph

        graph_data = [
            list(self._ends_iter_()),
            list(self._end_connections_iter_()),
        ]

        return Graph(data=graph_data, immutable=True)

    @cached_method
    def _components_(self):
        return tuple(tuple(cycle) for cycle in self._ends_graph_().cycle_basis())

    def number_of_components(self):
        return len(self._components_())

    def _orientations_iter_(self):
        from itertools import product

        cycles_oriented = [
            [cycle, tuple(reversed(cycle))] for cycle in self._components_()
        ]

        for (*cycles,) in product(*cycles_oriented):
            yield OrientedBridgeLink(self.bridge_number, self.braid, tuple(cycles))

    def orientations(self):
        return tuple(self._orientations_iter_())

    def _ascii_art_braid_(self):
        n = self.bridge_number * 2
        result = ""
        for ind, j in enumerate(reversed(self.braid)):
            j_pos = abs(j)
            result += "|   " * (j_pos - 1)
            result += " \\ / "
            result += "   |" * (n - j_pos - 1) + "\n"
            result += "|   " * (j_pos - 1)
            result += "  \\  " if j > 0 else "  /  "
            result += "   |" * (n - j_pos - 1) + "\n"
            result += "|   " * (j_pos - 1)
            result += " / \\ "
            result += "   |" * (n - j_pos - 1) + "\n"
            if ind != len(self.braid) - 1:
                if j_pos == abs(self.braid[-ind - 2]) + 1:
                    result += "   ".join("|" for i in range(j_pos - 1))
                    result += "   /   "
                    result += "   ".join("|" for i in range(n - j_pos)) + "\n"
                elif j_pos == abs(self.braid[-ind - 2]) - 1:
                    result += "   ".join("|" for i in range(j_pos))
                    result += "   \\   "
                    result += "   ".join("|" for i in range(n - j_pos - 1)) + "\n"
                else:
                    result += "   ".join("|" for i in range(n)) + "\n"

        return result

    def _ascii_art_cups_(self):
        result = "   ".join("|" for i in range(2 * self.bridge_number)) + "\n"
        result += "   ".join("\\___/" for _ in range(self.bridge_number)) + "\n"

        return result

    def _ascii_art_caps_(self):
        result = "   ".join(" ___ " for _ in range(self.bridge_number)) + "\n"
        result += "   ".join("/   \\" for _ in range(self.bridge_number)) + "\n"
        result += "   ".join("|" for i in range(2 * self.bridge_number)) + "\n"

        return result

    def ascii_art(self):
        result = self._ascii_art_caps_()
        result += self._ascii_art_braid_()
        result += self._ascii_art_cups_()

        return result


@dataclass(frozen=True, eq=False)
class OrientedBridgeLink(BridgeLink):
    cycles: tuple[tuple[StrandEnd]]

    def set_colors(self, *colors: WeightSpaceElement):
        """
        Makes a colored link.

        If there are more components than colors given,
        the colors will be repeated until all components are colored.
        In particular, if one color is given,
        all components get colored in this color.
        """
        return ColoredOrientedBridgeLink(
            self.bridge_number, self.braid, self.cycles, colors
        )

    def _oriented_cups_and_caps_iter_(self, cycle_number=None):
        """
        Iterates over pairs `(first_end, second_end)`
        corresponding to cups or caps.
        The orientation is `first_end -> second_end`.
        No specific order of pairs is guaranteed.

        If `cycle_number` is given, then will
        return only cups and caps from this cycle.
        """
        from itertools import cycle, islice

        if cycle_number is None:
            relevant_cycles = self.cycles
        else:
            relevant_cycles = [self.cycles[cycle_number]]

        for cycle_ in relevant_cycles:
            cycle_iterator = cycle(cycle_)
            # if the first two ends do not form
            # a cap or a cup, shift the cycle by one.
            if cycle_[0].is_top != cycle_[1].is_top:
                next(cycle_iterator)
            pairs_in_cycle = len(cycle_) // 2
            for _ in range(pairs_in_cycle):
                yield tuple(islice(cycle_iterator, 2))

    @cached_method
    def _top_ends_oriented_downwards_(self):
        # can be done more efficiently, but we don't need it
        relevant_ends = frozenset(
            end for _, end in self._oriented_cups_and_caps_iter_() if end.is_top
        )

        return relevant_ends

    @cached_method
    def _bottom_ends_oriented_downwards_(self):
        # can be done more efficiently, but we don't need it
        relevant_ends = frozenset(
            end for end, _ in self._oriented_cups_and_caps_iter_() if not end.is_top
        )

        return relevant_ends

    def _ascii_art_top_orientations_(self):
        result = "   ".join(
            "V" if end in self._top_ends_oriented_downwards_() else "|"
            for end in self._top_ends_iter_()
        )
        result += "\n"
        result += "   ".join("|" for i in range(2 * self.bridge_number)) + "\n"

        return result

    def _ascii_art_bottom_orientations_(self):
        result = "   ".join("|" for i in range(2 * self.bridge_number)) + "\n"
        result += "   ".join(
            "V" if end in self._bottom_ends_oriented_downwards_() else "|"
            for end in self._bottom_ends_iter_()
        )
        result += "\n"

        return result

    def ascii_art(self):
        result = self._ascii_art_caps_()
        result += self._ascii_art_top_orientations_()
        result += self._ascii_art_braid_()
        result += self._ascii_art_bottom_orientations_()
        result += self._ascii_art_cups_()

        return result


@dataclass(frozen=True, eq=False)
class ColoredOrientedBridgeLink(OrientedBridgeLink):
    """
    Oriented link with colors.

    Create with `set_colors` method of `OrientedBridgeLink`.
    """

    _color_data: tuple[WeightSpaceElement]

    @staticmethod
    def _involution_(Phi: RootSystem):
        automorphism_on_indices = Phi.cartan_type().opposition_automorphism()
        weight_lattice = Phi.weight_lattice()

        def on_basis(x):
            return weight_lattice.fundamental_weight(automorphism_on_indices[x])

        mor = weight_lattice.module_morphism(on_basis=on_basis, codomain=weight_lattice)

        return mor

    @staticmethod
    def _root_system_(color: WeightSpaceElement):
        return color.parent().root_system

    @classmethod
    def _opposite_color_(cls, color: WeightSpaceElement):
        return cls._involution_(cls._root_system_(color))(color)

    def _cycle_color_iter_(self):
        from itertools import cycle

        return zip(cycle(self._color_data), self._components_())

    def colors(self):
        return tuple(col for col, _ in self._cycle_color_iter_())

    @cached_method
    def _ends_colors_(self):
        """
        Colors the ends;

        Ends facing up receive the `color` of their cycle;
        Ends facing down get `self._opposite_color_(color)`
        applied the them.
        """
        ends_colors = {}
        for cycle_number, color in enumerate(self.colors()):
            opposite_color = self._opposite_color_(color)
            for end_prev, end_next in self._oriented_cups_and_caps_iter_(cycle_number):
                if end_prev.is_top:
                    ends_colors[end_prev] = color
                    ends_colors[end_next] = opposite_color
                else:
                    ends_colors[end_prev] = opposite_color
                    ends_colors[end_next] = color

        return MappingProxyType(ends_colors)

    def colors_top(self):
        return tuple(self._ends_colors_()[end] for end in self._top_ends_iter_())

    def colors_bottom(self):
        return tuple(self._ends_colors_()[end] for end in self._bottom_ends_iter_())

    def _cups_ebranes_iter_(
        self,
        reduced_cup: int | None = None,
    ):
        """
        Iterates over pieces of cup brane.

        If `reduced_cup` is given, the `i`th cup
        will be reduced. Enumeration starts with `0`.
        """
        from klrw.standard_ebranes import standard_ebranes, standard_reduced_ebranes

        for i in range(self.bridge_number):
            bot_end = StrandEnd(strand=2 * i + 1, is_top=False)
            color = self._ends_colors_()[bot_end]
            if i == reduced_cup:
                yield standard_reduced_ebranes(self._root_system_(color), color)
            else:
                yield standard_ebranes(self._root_system_(color), color)

    def _cups_brane_(
        self,
        reduced_cup: int | None = None,
    ):
        """
        Makes a cup brane.

        If `reduced_cup` is given, the `i`th cup
        will be reduced. Enumeration starts with `0`.
        """
        assert self.bridge_number >= 1

        ebrane_iter = self._cups_ebranes_iter_(reduced_cup)
        product_ebrane = next(ebrane_iter)
        for brane in ebrane_iter:
            product_ebrane @= brane

        return product_ebrane

    def _braided_cups_brane_(
        self,
        reduced_cup: int | None = None,
        verbose: bool = True,
    ):
        """
        Makes a cup brane braided with respect to link's `braid`.

        If `reduced_cup` is given, the `i`th cup
        will be reduced. Enumeration starts with `0`.
        """
        from klrw.braiding_functor import BraidingFunctor

        complex = self._cups_brane_(reduced_cup)
        for i in self.braid:
            if verbose:
                print(">>>", i, "<<<")

            complex = BraidingFunctor(i)(complex)

        return complex

    def _caps_brane_(
        self,
        reduced_cap: int | None = None,
    ):
        from klrw.simple_module import EBrane

        top_color_tuple = ()
        for i in range(self.bridge_number):
            top_end = StrandEnd(strand=2 * i + 1, is_top=True)
            color = self._ends_colors_()[top_end]
            top_color_tuple += (color,)

        return EBrane(*top_color_tuple, reduced_parts=(reduced_cap,))

    @cached_method
    def _link_homset_(self, reduced_cycle=None):
        from klrw.simple_module import KLRWPerfectComplexToEbrane_Homset

        if reduced_cycle is not None:
            cup, cap = list(
                self._oriented_cups_and_caps_iter_(cycle_number=reduced_cycle)
            )[:2]
            # we are not sure if the first one is a cup, not cap;
            # may need to swap
            if cup[0].is_top:
                cup, cap = cap, cup

            reduced_cup = (cup[0].strand - 1) // 2
            reduced_cap = (cap[0].strand - 1) // 2
        else:
            reduced_cup = None
            reduced_cap = None

        cup = self._braided_cups_brane_(reduced_cup)
        cap = self._caps_brane_(reduced_cap)

        return KLRWPerfectComplexToEbrane_Homset(cup, cap)

    def link_homology(self, base_ring=None, reduced_cycle=None):
        """
        Returns link homology.

        If `base_ring` is not given, computation is done over Integers.
        If `reduced_cycle` is given, computes reduced link invariant.
        """
        homset = self._link_homset_(reduced_cycle)

        return homset.homology(base_ring)

    def poincare_polynomial(self, base_field=None, reduced_cycle=None):
        """
        Returns Poincare polynomial of link homology.

        If `base_field` is not given, computation is done over Rational numbers.
        If `reduced_cycle` is given, computes reduced link invariant.
        """
        from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
        from sage.rings.all import ZZ, QQ

        if base_field is None:
            base_field = QQ
        assert base_field.is_field(), "base_field has to be a field"

        lau_poly = LaurentPolynomialRing(ZZ, 2, ["t", "q"])
        homology = self.link_homology(base_ring=base_field, reduced_cycle=reduced_cycle)
        t = lau_poly("t")
        q = lau_poly("q")
        poincare_polynomial = sum(
            homology_group.ngens() * t ** grading[0] * q ** grading[1]
            for grading, homology_group in homology.items()
        )

        return poincare_polynomial
