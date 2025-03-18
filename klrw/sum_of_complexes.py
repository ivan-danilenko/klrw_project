from collections import defaultdict
from types import MappingProxyType

from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.lazy_attribute import lazy_attribute
from sage.matrix.constructor import matrix

from klrw.free_complex import ComplexOfFreeModules, ShiftedComplexOfFreeModules
from klrw.perfect_complex import KLRWPerfectComplex


class SumOfComplexesOfFreeModules(ComplexOfFreeModules):
    @staticmethod
    def __classcall__(
        cls,
        *complexes,
        keep_subdivisions=True,
    ):
        instance = UniqueRepresentation.__classcall__(
            cls,
            *complexes,
        )
        instance.keep_subdivisions = keep_subdivisions
        return instance

    def __init__(
        self,
        *complexes,
    ):
        from klrw.misc import get_from_all_and_assert_equality

        assert all(
            isinstance(comp, ComplexOfFreeModules | ShiftedComplexOfFreeModules)
            for comp in complexes
        )
        assert complexes, "Need at least one complex"
        self._ring = get_from_all_and_assert_equality(
            lambda x: x.ring(),
            complexes,
        )

        self._parts = complexes

    @lazy_attribute
    def _grading_group(self):
        from klrw.misc import get_from_all_and_assert_equality

        return get_from_all_and_assert_equality(
            lambda x: x.shift_group(),
            self._parts,
        )

    def component_rank(self, grading):
        return sum(comp.component_rank(grading) for comp in self._parts)

    def component_rank_iter(self):
        result = defaultdict(int)
        for comp in self._parts:
            for grading, dim in self._domain_shifted.component_rank_iter():
                result[grading] += comp.component_rank(grading)

        yield from result.items()

    def subdivisions(self, grading, position=None):
        grading = self.hom_grading_group()(grading)
        if grading in self._subdivisions:
            if position is None:
                return self._subdivisions[grading]
            else:
                return self._subdivisions[grading][position]

        if position is None:
            return tuple((0,))
        assert position == 0
        return 0

    @lazy_attribute
    def _subdivisions(self):
        subdivisions = {}
        for a, comp in enumerate(self._parts):
            for hom_deg, rank in comp.component_rank_iter():
                if hom_deg not in subdivisions:
                    subdivisions[hom_deg] = [
                        0,
                    ] * (len(self._parts) + 1)
                # record only current dimension;
                # will make the sequence cummmulative later
                subdivisions[hom_deg][a + 1] = rank

        # make subdivisions cummulative
        for combined_hom_degree, subdiv in subdivisions.items():
            for a in range(len(subdiv) - 1):
                subdiv[a + 1] += subdiv[a]

        return MappingProxyType(subdivisions)

    @lazy_attribute
    def differential(self):
        from klrw.misc import get_from_all_and_assert_equality

        diff_degree = get_from_all_and_assert_equality(
            lambda x: x.differential.degree(),
            self._parts,
        )
        sign = self._parts[0].get_from_all_and_assert_equality(
            lambda x: x.differential.sign,
            self._parts,
        )
        diff_hom_degree = diff_degree.homological_part()

        gradings = frozenset()
        differential = {}
        for comp in self._parts:
            gradings |= frozenset(comp.differential.support())

        for grading in gradings:
            next_grading = grading + diff_hom_degree
            row_subdivisions = [0]
            col_subdivisions = [0]
            for comp in self._parts:
                col_subdivisions.append(
                    col_subdivisions[-1] + comp.component_rank(grading)
                )
                row_subdivisions.append(
                    row_subdivisions[-1] + comp.component_rank(next_grading)
                )

            differential_component = matrix(
                self.KLRW_algebra().opposite,
                ncols=col_subdivisions[-1],
                nrows=row_subdivisions[-1],
                sparse=True,
            )

            for i in range(len(self._parts)):
                # set_block does not take into accout sparsity
                iterator = self._parts[i].differential(grading).dict(copy=False).items()
                for (a, b), entry in iterator:
                    indices = (row_subdivisions[i] + a, col_subdivisions[i] + b)
                    differential_component[indices] = entry
                # differential_component.set_block(
                #     row=row_subdivisions[i],
                #     col=col_subdivisions[i],
                #     block=self._parts[i].differential(grading),
                # )
            if self.keep_subdivisions:
                differential_component._subdivisions = (
                    row_subdivisions,
                    col_subdivisions,
                )
            differential_component.set_immutable()
            differential[grading] = differential_component

        return self.DifferentialClass(
            underlying_module=self,
            differential_data=differential,
            degree=diff_degree,
            sign=sign,
            check=True,
        )


class SumOfKLRWPerfectComplexes(SumOfComplexesOfFreeModules, KLRWPerfectComplex):
    @lazy_attribute
    def _KLRW_algebra(self):
        from klrw.misc import get_from_all_and_assert_equality

        return get_from_all_and_assert_equality(
            lambda x: x.KLRW_algebra(),
            self._parts,
        )

    @lazy_attribute
    def _extended_grading_group(self):
        from klrw.misc import get_from_all_and_assert_equality

        return get_from_all_and_assert_equality(
            lambda x: x.shift_group(),
            self._parts,
        )

    @lazy_attribute
    def _projectives(self):
        projectives = defaultdict(list)
        for comp in self._parts:
            for grading, projs in comp.projectives_iter():
                projectives[grading] += projs

        return MappingProxyType(projectives)
