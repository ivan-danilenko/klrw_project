from typing import Iterable
from collections import defaultdict
from types import MappingProxyType

from sage.matrix.matrix0 import Matrix
from sage.matrix.constructor import matrix

from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.lazy_attribute import lazy_attribute, lazy_class_attribute

from .free_complex import (
    Graded_Homomorphism,
    GradedFreeModule,
    ShiftedGradedFreeModule,
    GradedFreeModule_Homset,
    GradedFreeModule_Homomorphism,
    Differential,
    ComplexOfFreeModules,
    ShiftedComplexOfFreeModules,
)
from .perfect_complex import (
    KLRWDirectSumOfProjectives,
    ShiftedKLRWDirectSumOfProjectives,
    KLRWDirectSumOfProjectives_Homset,
    KLRWPerfectComplex,
    ShiftedKLRWPerfectComplex,
    KLRWPerfectComplex_Homset,
    KLRWDifferential,
)
from .gradings import (
    HomologicalGradingGroupElement,
    ExtendedQuiverGradingGroupElement,
)


class MulticomplexOfFreeModules(GradedFreeModule):
    """
    A bi-, tri-, etc, complex of graded free modules.

    This should be usually created by tensor products,
    derived functors, etc.
    """

    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedMulticomplexOfFreeModules

    @lazy_class_attribute
    def HomsetClass(cls):
        # We don't want to check commuting with all
        # differentials, so we just use graded homsets.
        return GradedFreeModule_Homset

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        assert issubclass(cls.HomsetClass, GradedFreeModule_Homset)
        return GradedFreeModule_Homset

    @lazy_class_attribute
    def DifferentialClass(cls):
        return Differential

    @staticmethod
    def __classcall__(
        cls,
        ring,
        differentials: Iterable[
            dict[
                HomologicalGradingGroupElement | ExtendedQuiverGradingGroupElement,
                Matrix | dict,
            ]
            | GradedFreeModule_Homomorphism
        ],
        signs: Iterable,
        differential_degrees: Iterable[
            HomologicalGradingGroupElement | ExtendedQuiverGradingGroupElement
        ],
        **kwargs,
    ):
        # to make hashable
        differentials = tuple(
            Graded_Homomorphism(
                None,
                diff,
                copy=False,
            )
            for diff in differentials
        )
        for diff in differentials:
            diff.set_immutable()

        return super().__classcall__(
            cls,
            ring=ring,
            differentials=differentials,
            signs=signs,
            differential_degrees=differential_degrees,
            **kwargs,
        )

    def __init__(
        self,
        differentials,
        signs,
        differential_degrees,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._differentials = tuple(
            self.DifferentialClass(
                underlying_module=self,
                differential_data=differentials[i],
                sign=signs[i],
                degree=differential_degrees[i],
            )
            for i in range(len(differentials))
        )
        self._check_differentials_()

    def _check_differentials_(self, differentials=None):
        if differentials is None:
            differentials = self._differentials

        for i, di in enumerate(differentials):
            for dj in differentials[i + 1 :]:
                assert di * dj == dj * di
            for j, dj in enumerate(differentials):
                if i != j:
                    assert dj.sign(di.degree()) == 1

    def differentials(self, n: int | None = None):
        """
        Return `n`-th differential.
        """
        if n is None:
            return self._differentials
        return self._differentials[n]

    def __repr__(self):
        from pprint import pformat

        result = "A complex of free modules with graded dimension\n"
        result += pformat(dict(self.component_rank_iter()))
        result += "\nwith differentials\n"
        for i, diff in enumerate(self.differentials):
            result += "===" + repr(i) + "===\n"
            for grading in sorted(diff.support()):
                result += repr(grading) + " -> " + repr(grading + diff.hom_degree())
                result += ":\n"
                result += repr(diff(grading)) + "\n"

        return result

    def totalization(self):
        return TotalizationOfMulticomplexOfFreeModules(self)


class ShiftedMulticomplexOfFreeModules(ShiftedGradedFreeModule):
    """
    A shifted bi-, tri-, etc, complex of graded free modules.
    """

    @lazy_class_attribute
    def OriginalClass(cls):
        return MulticomplexOfFreeModules

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        return cls.OriginalClass.HomsetClassWithoutDifferential

    def differential(self, n: int):
        """
        Return `n`-th differential.
        """
        return self.original.differential(n)[self.shift]

    def totalization(self):
        original_totalization = self.original.totalization()
        shift = original_totalization.totalization_morphism(self.shift)
        return original_totalization[shift]


class TotalizationOfGradedFreeModule(GradedFreeModule):
    """
    Totalize gradings.

    Some gradings become the same.
    This is done by the attribute `totalization_morphism`
    of the shift group and hom grading group of `graded_module`.
    """

    """
    @lazy_class_attribute
    def OriginalClass(cls):
        return cls

    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedTotalizationOfGradedFreeModule
    """

    @staticmethod
    def __classcall__(
        cls,
        graded_module: GradedFreeModule,
    ):
        return UniqueRepresentation.__classcall__(
            cls,
            graded_module,
        )

    def __init__(
        self,
        graded_module: GradedFreeModule,
    ):
        self._before_totalization = graded_module
        self._ring = graded_module.ring()

        self.totalization_morphism = graded_module.shift_group().totalization_morphism
        hom_grading_group = graded_module.hom_grading_group()
        self.hom_totalization_morphism = hom_grading_group.totalization_morphism
        self._grading_group = self.hom_totalization_morphism.codomain()

        self.grading_preimages = defaultdict(list)
        self._indices_within_total_grading = {}
        self._subdivisions = defaultdict(lambda: [0])
        for hom_grading, rank in graded_module.component_rank_iter():
            total_grading = self.hom_totalization_morphism(hom_grading)
            self._indices_within_total_grading[hom_grading] = len(
                self.grading_preimages[total_grading]
            )
            self.grading_preimages[total_grading].append(hom_grading)
            subdiv_in_grading = self._subdivisions[total_grading]
            subdiv_in_grading.append(subdiv_in_grading[-1] + rank)

    def indices_within_total_grading(self, grading):
        return self._indices_within_total_grading[grading]

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

    def component_rank(self, grading):
        if grading in self._subdivisions:
            return self._subdivisions[grading][-1]
        else:
            return 0

    def component_rank_iter(self):
        for grading, subdiv in self._subdivisions.items():
            yield (grading, subdiv[-1])

    # def base_change(self, other):
    #     graded_module = self._reduction[1][0]
    #     return self._replace_(
    #         multicomplex=graded_module.base_change(other),
    #     )

    @staticmethod
    def totalize_morphism(morphism, sign=None):
        """
        Totalize a morphism between two graded modules with several hom gradings.

        `sign` is a function of of hom degrees with values +1/-1.
        By default, it's constantly 1.
        """
        from klrw.free_complex import ShiftedObject

        original_domain, domain_shift = ShiftedObject.original_and_shift(
            morphism.domain()
        )
        original_codomain, codomain_shift = ShiftedObject.original_and_shift(
            morphism.codomain()
        )
        domain_hom_shift = domain_shift.homological_part()
        codomain_hom_shift = codomain_shift.homological_part()

        domain = original_domain.totalization()
        codomain = original_codomain.totalization()
        assert domain.shift_group() == codomain.shift_group()
        assert domain.hom_grading_group() == codomain.hom_grading_group()
        hom_totalization_morphism = domain.hom_totalization_morphism
        total_domain_hom_shift = hom_totalization_morphism(domain_hom_shift)
        total_codomain_hom_shift = hom_totalization_morphism(codomain_hom_shift)

        totalized_morphism_dict = defaultdict(dict)
        for hom_degree, map in morphism:
            # we need to use the original gradings, not totalized ones.
            # this is why we do shifts by hands, instead of working
            # with shifted objects.
            total_degree = hom_totalization_morphism(hom_degree)
            domain_index = domain.indices_within_total_grading(
                hom_degree + domain_hom_shift
            )
            codomain_index = codomain.indices_within_total_grading(
                hom_degree + codomain_hom_shift
            )
            column_subdivision = domain.subdivisions(
                total_degree + total_domain_hom_shift, domain_index
            )
            row_subdivision = codomain.subdivisions(
                total_degree + total_codomain_hom_shift, codomain_index
            )
            if sign is not None:
                _sign = sign(hom_degree)
            else:
                _sign = 1

            # new_diff_dict_of_dicts[domain_total_degree] = {}
            totalized_morphism_component = totalized_morphism_dict[total_degree]
            for (a, b), entry in map.dict(copy=False).items():
                indices = (a + row_subdivision, b + column_subdivision)
                totalized_morphism_component[indices] = _sign * entry

        domain = domain[domain.totalization_morphism(domain_shift)]
        codomain = codomain[codomain.totalization_morphism(codomain_shift)]

        return domain.hom(
            codomain, totalized_morphism_dict, ignore_checks=frozenset(["chain"])
        )


"""
class ShiftedTotalizationOfGradedFreeModule(ShiftedGradedFreeModule):
    @lazy_class_attribute
    def OriginalClass(cls):
        return TotalizationOfGradedFreeModule

    @lazy_attribute
    def _shift_before_totalization(self):
        return self.original._befor

    def indices_within_total_grading(self, grading):
        return self.original.indices_within_total_grading(grading + self.hom_shift())
"""


class TotalizationOfMulticomplexOfFreeModules(
    TotalizationOfGradedFreeModule, ComplexOfFreeModules
):
    """
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedTotalizationOfGradedFreeModule
    """

    @lazy_attribute
    def differential(self):
        from klrw.misc import get_from_all_and_assert_equality
        from klrw.free_complex import ShiftedObject

        self.original_signs = [
            diff.sign for diff in self._before_totalization.differentials()
        ]

        multicomplex = self._before_totalization
        # we totalize morphisms and multuply blocks by appropriate signs.
        totalized_differentials = [
            self.totalize_morphism(
                differential,
                sign=lambda hom_degree: self.grading_sign(i, hom_degree),
            )
            for i, differential in enumerate(multicomplex.differentials())
        ]
        # Checking that codomains are the same.
        # Mostly, that the shifts are correct.
        try:
            codomain = get_from_all_and_assert_equality(
                lambda diff: diff.codomain(),
                totalized_differentials,
            )
        except AssertionError:
            raise ValueError("Differential degrees differ after totalization.")
        differential_degree = ShiftedObject.find_shift(self, codomain)

        new_differential = sum(totalized_differentials)

        return self.DifferentialClass(
            underlying_module=self,
            differential_data=new_differential,
            degree=differential_degree,
            check=True,
        )

    def grading_sign(self, i: int, grading):
        from sage.misc.misc_c import prod

        return prod((self.original_signs[j](grading) for j in range(i)))


"""
class ShiftedTotalizationOfMulticomplexOfFreeModules(
    ShiftedTotalizationOfGradedFreeModule, ShiftedComplexOfFreeModules
):
    @lazy_class_attribute
    def OriginalClass(cls):
        return TotalizationOfGradedFreeModule
"""


class KLRWPerfectMulticomplex(MulticomplexOfFreeModules, KLRWDirectSumOfProjectives):
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedKLRWPerfectMulticomplex

    @lazy_class_attribute
    def HomsetClass(cls):
        # We don't want to check commuting with all
        # differentials, so we just use graded homsets.
        return KLRWDirectSumOfProjectives_Homset

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        assert issubclass(cls.HomsetClass, KLRWDirectSumOfProjectives_Homset)
        return KLRWDirectSumOfProjectives_Homset

    @lazy_class_attribute
    def DifferentialClass(cls):
        return KLRWDifferential

    def __repr__(self):
        from pprint import pformat

        result = "A complex of projective modules\n"
        result += pformat(dict(self.projectives()))
        result += "\nwith differentials\n"
        for i, diff in enumerate(self.differentials()):
            result += "===" + repr(i) + "===\n"
            for grading in sorted(diff.support()):
                result += repr(grading) + " -> " + repr(grading + diff.hom_degree())
                result += ":\n"
                result += repr(diff(grading)) + "\n"

        return result

    def totalization(self):
        return TotalizationOfKLRWPerfectMulticomplex(self)


class ShiftedKLRWPerfectMulticomplex(
    ShiftedMulticomplexOfFreeModules, ShiftedKLRWDirectSumOfProjectives
):
    @lazy_class_attribute
    def OriginalClass(cls):
        return KLRWPerfectMulticomplex


class TotalizationOfKLRWPerfectMulticomplex(
    TotalizationOfMulticomplexOfFreeModules, KLRWPerfectComplex
):
    def __init__(
        self,
        multicomplex: KLRWPerfectMulticomplex,
    ):
        totalization_morphism = multicomplex.shift_group().totalization_morphism
        self._extended_grading_group = totalization_morphism.codomain()
        self._KLRW_algebra = multicomplex.KLRW_algebra()
        TotalizationOfGradedFreeModule.__init__(self, multicomplex)

    @lazy_attribute
    def _projectives(self):
        projectives_dict_of_lists = {}
        for hom_grading, projs in self._before_totalization.projectives_iter():
            total_grading = self.hom_totalization_morphism(hom_grading)
            if total_grading not in projectives_dict_of_lists:
                number_of_pieces = len(self.subdivisions(total_grading)) - 1
                projectives_dict_of_lists[total_grading] = [None] * number_of_pieces
            index = self.indices_within_total_grading(hom_grading)
            projectives_dict_of_lists[total_grading][index] = projs

        self._projectives = defaultdict(list)
        for total_grading, list_of_projs in projectives_dict_of_lists.items():
            for projs in list_of_projs:
                self._projectives[total_grading] += projs

        return MappingProxyType(self._projectives)

    @lazy_class_attribute
    def HomsetClass(cls):
        # We don't want to check commuting with all
        # differentials, so we just use graded homsets.
        return KLRWPerfectComplex_Homset

    @lazy_class_attribute
    def HomsetClassWithoutDifferential(cls):
        assert issubclass(cls.HomsetClass, KLRWDirectSumOfProjectives_Homset)
        return KLRWDirectSumOfProjectives_Homset

    @lazy_class_attribute
    def DifferentialClass(cls):
        return KLRWDifferential

    def __repr__(self):
        return KLRWPerfectComplex.__repr__(self)
