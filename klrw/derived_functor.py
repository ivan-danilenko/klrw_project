from typing import Callable, Iterable
from collections import defaultdict

from sage.structure.unique_representation import UniqueRepresentation
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_method

from klrw.perfect_complex import (
    KLRWIrreducibleProjectiveModule,
    KLRWPerfectComplex,
    ShiftedKLRWPerfectComplex,
    CenterHomMultiplication,
)


class KLRWDerivedFunctor(UniqueRepresentation):
    def __init__(
        self,
        # domain_klrw_algebra: KLRWAlgebra,
        # codomain_klrw_algebra: KLRWAlgebra,
        on_projectives: Callable,
        on_crossings: Callable,
        on_dots: Callable,
        # differential_degree: HomologicalGradingGroupElement,
        # extra_homological_grading_group: HomologicalGradingGroup = None,
        on_parameters: Callable | None = None,
    ):
        # self.domain_klrw_algebra = domain_klrw_algebra
        # self.codomain_klrw_algebra = codomain_klrw_algebra
        self.on_projectives = on_projectives
        self.on_crossings = on_crossings
        self.on_dots = on_dots
        # if extra_homological_grading_group is None:
        #     extra_homological_grading_group = HomologicalGradingGroup(
        #         R=ZZ,
        #         homological_grading_names=(None,),
        #     )
        # self.extra_homological_grading_group = extra_homological_grading_group
        # self.differential_degree = extra_homological_grading_group(
        #     differential_degree
        # )
        if on_parameters is None:

            def on_parameters(coeff):
                return coeff

        #    domain_parameters = domain_klrw_algebra.base().without_dots
        #    codomain_parameters = codomain_klrw_algebra.base().without_dots
        #    on_parameters = domain_parameters.convert_map_from(codomain_parameters)
        self.on_parameters = on_parameters

    def __call__(self, argument):
        if isinstance(argument, KLRWPerfectComplex | ShiftedKLRWPerfectComplex):
            return self.on_complex(argument)

        raise ValueError(
            "Don't know how to apply derived functors to {}".format(argument)
        )

    @cached_method
    def braid_to_chain_map(self, braid, domain_klrw_algebra, representative=False):
        right_state = braid.right_state()
        word = braid.word()
        if not word:
            right_proj = KLRWIrreducibleProjectiveModule(state=right_state)
            right = self.on_projectives(right_proj, domain_klrw_algebra)
            return right.hom_set(right).one().homology_class()

        result = self.on_crossings(word[-1], right_state, domain_klrw_algebra)
        intermediate_state = right_state

        # iterating in reverse order over all
        # elements except the last one,
        # since it's already taken into account.
        for i in range(len(word) - 2, -1, -1):
            intermediate_state = intermediate_state.act_by_s(word[i + 1])
            result = result * self.on_crossings(
                word[i], intermediate_state, domain_klrw_algebra
            )

        if representative:
            return result.representative()
        return result

    """
    def dot_to_chain_map(self, dot_index: DotVariableIndex, state):
        eq_grading_group = self.equivariant_grading_group()
        equivariant_shift = eq_grading_group.dot_algebra_grading(dot_index)
        codomain = self[0, equivariant_shift]

        standard_dot_index = self.dot_to_standard(dot_index)
        chain_map_dict = {}
        if standard_dot_index is None:
            coeff = self.KLRW_algebra().base().variables[dot_index].monomial
            for hom_deg, projs in self.standard_complex.projectives_iter():
                chain_map_dict[hom_deg] = {
                    (j, j): self.KLRW_algebra().term(
                        self.KLRW_algebra().braid(
                            state=self.state_from_standard(projs[j].state),
                            word=(),
                        ),
                        coeff=coeff,
                    )
                    for j in range(len(projs))
                }
        else:
            standard_chain = self.standard_complex._braided_dot_morphism_(
                dot_color=standard_dot_index.vertex,
                dot_position=standard_dot_index.number,
                self_on_right=True,
            )
            for hom_deg, map in standard_chain:
                chain_map_dict[hom_deg] = {
                    (i, j): self.element_from_standard(entry.value)
                    for (i, j), entry in map.dict(copy=False).items()
                }

        chain_map = self.hom(codomain, chain_map_dict)
        return chain_map.homology_class()
    """

    def _on_dot_monomial_(self, dots_items: Iterable, state, domain_klrw_algebra):
        from sage.misc.misc_c import prod

        dots_dict = {x: y for x, y in dots_items}

        return prod(
            (
                self.on_dots(dot_index, state, domain_klrw_algebra) ** pow
                for dot_index, pow in dots_dict.items()
            ),
        )

    def on_klrw_elements(self, element, representative=False):
        """
        Makes a chain map corresponding to a KLRW element.

        Any KLRW element with fixed left and right states
        is a morphism between projectives.
        Gives the image of this map in Ext^0 between
        the images of projectives.
        If `representative=True` returns a representative
        of this ext class.
        """
        left_state = element.left_state(check_if_all_have_same_left_state=True)
        dot_algebra = None
        result = None
        for braid, coeff in element:
            braid_part = self.braid_to_chain_map(braid, element.parent())
            if dot_algebra is None:
                dot_algebra = braid_part.domain().KLRW_algebra().base()

            coeff_part = None
            for cent_poly, dict_of_dots in dot_algebra._dict_of_dots_iterator_(coeff):
                if dict_of_dots:
                    dot_part = self._on_dot_monomial_(
                        frozenset(dict_of_dots.items()),
                        left_state,
                        element.parent(),
                    )
                    """
                    dot_part = prod(
                        (
                            self.on_dots(dot_index, left_state, element.parent()) ** pow
                            for dot_index, pow in dict_of_dots.items()
                        ),
                    )
                    """
                else:
                    left_proj = KLRWIrreducibleProjectiveModule(state=left_state)
                    left = self.on_projectives(left_proj, element.parent())
                    dot_part = left.hom_set(left).one().homology_class()

                cent_poly = self.on_parameters(cent_poly)
                # We could just write
                # `piece = cent_poly * dot_part`
                # but this will trigger checking is `cent_poly` is central.
                # So we do the same, but explicitly pass `check=False`
                # to `_act_`
                cent_hom_mult = CenterHomMultiplication(
                    other=cent_poly.parent(),
                    hom_set=dot_part.parent(),
                )
                piece = cent_hom_mult._act_(
                    cent_poly,
                    dot_part,
                    check=False,
                )

                if coeff_part is None:
                    coeff_part = piece
                else:
                    coeff_part += piece

            # if we read braid as a map from left to right,
            # when dots act first because they are on the left
            # and the map that acts first should be on the right
            if result is None:
                result = braid_part * coeff_part
            else:
                result += braid_part * coeff_part

        if representative:
            return result.representative()
        return result

    # def on_sum_of_projectives(
    #   self,
    #   argument: KLRWDirectSumOfProjectives | ShiftedKLRWDirectSumOfProjectives,
    # ):
    #    if isinstance(argument, KLRWDirectSumOfProjectives):
    #        return ImagePerfectComplex(self, argument)
    #    elif isinstance(argument, ShiftedKLRWDirectSumOfProjectives):
    #        original_image = self.on_sum_of_projectives(argument.original)
    #        embedding = original_image.shift_group().summand_embedding(0)
    #        new_shift = embedding(argument.shift)
    #        return original_image[new_shift]
    #    else:
    #        raise ValueError("{} is not a complex.".format(argument))

    def on_chain_map(self, chain_map, **kwargs):
        raise NotImplementedError("Need to fix order of iterated cones")
        # image_domain = self(chain_map.domain(), totalize=False)
        # image_codomain = self(chain_map.codomain(), totalize=False)
        #
        # image_chain_map_dict_of_dicts = defaultdict(dict)
        # for hom_deg, map in chain_map:
        #    for (i, j), entry in map.dict(copy=False).items():
        #        entry_image = self.on_klrw_elements(entry.value)
        #        for extra_hom_deg, image_piece in entry_image:
        #            combined_hom_degree = image_domain.combined_hom_degree(
        #               hom_deg,
        #               extra_hom_deg,
        #            )
        #            image_comp = image_chain_map_dict_of_dicts[combined_hom_degree]
        #            row_subdiv = image_codomain.subdivisions(combined_hom_degree, i)
        #            column_subdiv = image_domain.subdivisions(combined_hom_degree, j)
        #            # setting a block in the matrix
        #            for (a, b), elem in image_piece.dict(copy=False).items():
        #                image_comp[row_subdiv + a, column_subdiv + b] = elem
        #
        # image_chain_map_dict = {}
        # for combined_hom_degree, comp_dict in image_chain_map_dict_of_dicts.items():
        #    image_chain_map_dict[combined_hom_degree] = matrix(
        #        self.codomain_klrw_algebra.opposite,
        #        image_codomain.component_rank(combined_hom_degree),
        #        image_domain.component_rank(combined_hom_degree),
        #        comp_dict,
        #        sparse=True,
        #        immutable=True,
        #    )
        #    image_chain_map_dict[combined_hom_degree]._subdivisions = (
        #        image_codomain.subdivisions(combined_hom_degree),
        #        image_domain.subdivisions(combined_hom_degree)
        #    )
        #
        # return image_domain.hom(image_codomain, image_chain_map_dict, **kwargs)

    @staticmethod
    def _chains_iterator_(complex_):
        """
        Yields chains of homological grading.

        Split homological gradings into connected components.
        We say that two homological gradings are connected if
        there is a non-zero differential between them.
        Ordering in chains is such that the next grading
        has a map from the preceeding grading
        """
        from collections import deque

        diff_hom_deg = complex_.differential.hom_degree()
        degrees_left = set(complex_.gradings())
        while degrees_left:
            deg = degrees_left.pop()
            chain = deque([deg])
            next_deg = deg + diff_hom_deg
            while (
                next_deg - diff_hom_deg in complex_.differential.support()
                and next_deg in degrees_left
            ):
                degrees_left.remove(next_deg)
                chain.append(next_deg)
                next_deg += diff_hom_deg
            prev_deg = deg - diff_hom_deg
            while (
                prev_deg in complex_.differential.support() and prev_deg in degrees_left
            ):
                degrees_left.remove(prev_deg)
                chain.appendleft(prev_deg)
                prev_deg -= diff_hom_deg
            yield chain

    def _on_list_of_projectives_(self, domain_klrw_algebra, *projectives):
        terms = list(
            self.on_projectives(proj, domain_klrw_algebra) for proj in projectives
        )
        return KLRWPerfectComplex.sum(*terms)

    def _on_component_of_morphism_(
        self,
        domain: list[KLRWIrreducibleProjectiveModule] | KLRWPerfectComplex,
        codomain: list[KLRWIrreducibleProjectiveModule] | KLRWPerfectComplex,
        morphism_component,
        **kwargs
    ):
        """
        Make a morphism from a matrix.

        `domain` and `codomain` can be the lists of projectives,
        or can be the complexes representing their images.
        """
        if not isinstance(domain, KLRWPerfectComplex | ShiftedKLRWPerfectComplex):
            domain_klrw_algebra = morphism_component.parent().base_ring().algebra
            domain = self._on_list_of_projectives_(*domain, domain_klrw_algebra)
        if not isinstance(codomain, KLRWPerfectComplex | ShiftedKLRWPerfectComplex):
            domain_klrw_algebra = morphism_component.parent().base_ring().algebra
            codomain = self._on_list_of_projectives_(*codomain, domain_klrw_algebra)

        image_component_dict_of_dicts = defaultdict(dict)
        for (i, j), entry in morphism_component.dict(copy=False).items():
            entry_image = self.on_klrw_elements(entry.value)
            for hom_deg, image_piece in entry_image:
                row_subdiv = codomain.subdivisions(hom_deg, i)
                column_subdiv = domain.subdivisions(hom_deg, j)
                # setting a block in the matrix
                for (a, b), elem in image_piece.dict(copy=False).items():
                    image_piece = image_component_dict_of_dicts[hom_deg]
                    image_piece[row_subdiv + a, column_subdiv + b] = elem

        image_component_dict = {
            hom_deg: matrix(
                domain.KLRW_algebra().opposite,
                ncols=domain.component_rank(hom_deg),
                nrows=codomain.component_rank(hom_deg),
                entries=matrix_data,
                sparse=True,
            )
            for hom_deg, matrix_data in image_component_dict_of_dicts.items()
        }

        return domain.hom(codomain, image_component_dict, check=False, **kwargs)

    def on_complex(self, complex_: KLRWPerfectComplex | ShiftedKLRWPerfectComplex):
        from klrw.cones import KLRWIteratedCone

        # chains record homological degrees
        # in a piece
        # `-> P_{deg - diff_deg} -> P_{deg} -> P_{deg + diff_deg} -> `
        # into a deque with piece
        # `..., deg + diff_deg, deg, deg - diff_deg, ...`
        # Then we take iterated cones
        direct_summands = []
        for chain in self._chains_iterator_(complex_):
            print(*((hom_deg, len(complex_.projectives(hom_deg))) for hom_deg in chain))
            complexes = []
            for degree in chain:
                # define next complex and the morphism to it
                next_complex = [
                    self.on_projectives(proj, complex_.KLRW_algebra())
                    for proj in complex_.projectives(degree)
                ]
                complexes.append(next_complex)
            last_degree = chain.pop()
            morphisms = []
            # now chain doesn't have the last element because of pop()
            for degree in chain:
                differential_component = complex_.differential(degree)
                next_morphism = {
                    (i, j): self.on_klrw_elements(entry.value, representative=True)
                    for (i, j), entry in differential_component.dict(copy=False).items()
                }
                morphisms.append(next_morphism)
            # We solve for homotopies by iterations only
            # if we don't have invertible parameters.
            complex_summand = KLRWIteratedCone(morphisms, complexes, cache_level=1)
            """
            initial_degree = chain.popleft()
            # we make the image of the chain ignoring
            # the shift at first
            # and then shift it all at the very end.
            complex_piece = self._on_list_of_projectives_(
                complex_.KLRW_algebra(),
                *complex_.projectives(initial_degree),
            )
            if chain:
                # If there is a second term, make the cone.
                next_degree = chain.popleft()
                next_term = self._on_list_of_projectives_(
                    complex_.KLRW_algebra(),
                    *complex_.projectives(next_degree),
                )
                morphism = self._on_component_of_morphism_(
                    domain=next_term,
                    codomain=complex_piece,
                    morphism_component=complex_.differential(next_degree),
                )
                complex_piece = morphism.cone()
            while chain:
                # If there are more terms, do further cones.
                # In this case, we might need to correct
                # the morphism by a homotopy.
                previous_term = next_term
                prev_degree = next_degree
                next_degree = chain.popleft()
                print(prev_degree, "->", next_degree)
                next_term = self._on_list_of_projectives_(
                    complex_.KLRW_algebra(),
                    *complex_.projectives(next_degree),
                )
                morphism = self._on_component_of_morphism_(
                    domain=next_term,
                    codomain=previous_term,
                    morphism_component=complex_.differential(next_degree),
                )
                try:
                    morphism = complex_piece.lift_map(morphism, self_as_domain=False)
                except AssertionError as e:
                    from pickle import dump

                    with open("./pickles/problematic_complex.pickle", "wb") as f:
                        dump(complex_, file=f)
                    raise e
                complex_piece = morphism.cone()
            """
            # now take into account the homological shift of the initial term
            shift = -complex_.shift_group().from_homological_part(last_degree)
            complex_summand = complex_summand[shift]
            direct_summands.append(complex_summand)

        return KLRWPerfectComplex.sum(*direct_summands)


"""
class ImageDirectSumOfProjectives(KLRWDirectSumOfProjectives):
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedImageDirectSumOfProjectives

    @staticmethod
    def __classcall__(
        cls,
        derived_functor,
        preimage,
    ):
        return UniqueRepresentation.__classcall__(
            cls,
            derived_functor,
            preimage,
        )

    def __init__(
        self,
        derived_functor: KLRWDerivedFunctor,
        preimage,
    ):
        self.derived_functor = derived_functor
        self.preimage = preimage
        self._KLRW_algebra = derived_functor.codomain_klrw_algebra

    def subdivisions(self, grading=None, position=None):
        if grading is None:
            assert position is None
            return self._subdivisions
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
        for hom_deg, projs in self.preimage.projectives_iter():
            for a, pr in enumerate(projs):
                image_of_pr = self.derived_functor.on_projectives(pr.state)[
                    0, pr.equivariant_degree
                ]
                for extra_hom_deg, projs_in_image in image_of_pr.projectives_iter():
                    combined_hom_degree = self.combined_hom_degree(
                        hom_deg, extra_hom_deg
                    )
                    if combined_hom_degree not in subdivisions:
                        subdivisions[combined_hom_degree] = [
                            0,
                        ] * (len(projs) + 1)
                    # record only current dimension;
                    # will make the sequence cummmulative later
                    subdivisions[combined_hom_degree][a + 1] = len(projs_in_image)

        # make subdivisions cummulative
        for combined_hom_degree, subdiv in subdivisions.items():
            for a in range(len(subdiv) - 1):
                subdiv[a + 1] += subdiv[a]

        return MappingProxyType(subdivisions)

    def component_rank(self, grading):
        if grading in self._subdivisions:
            return self._subdivisions[grading][-1]
        else:
            return 0

    def component_rank_iter(self):
        for grading, subdiv in self._subdivisions.items():
            yield (grading, subdiv[-1])

    @lazy_attribute
    def _extended_grading_group(self):
        return (
            self.preimage.shift_group()
            + self.derived_functor.extra_homological_grading_group
        )

    @lazy_attribute
    def embedding_of_old_homological_part(self):
        return self.shift_group().homological_part.summand_embedding(0)

    @lazy_attribute
    def embedding_of_new_homological_part(self):
        return self.shift_group().homological_part.summand_embedding(1)

    def _new_sign(self):
        return self.shift_group().homological_part.default_sign_from_part(1)

    def combined_hom_degree(self, old_hom_deg, extra_hom_deg):
        return self.embedding_of_old_homological_part(
            old_hom_deg
        ) + self.embedding_of_new_homological_part(extra_hom_deg)

    @lazy_attribute
    def _projectives(self):
        # subdivisions for block structure of matrices
        projectives = defaultdict(list)
        for hom_deg, projs in self.preimage.projectives_iter():
            for a, pr in enumerate(projs):
                image_of_pr = self.derived_functor.on_projectives(pr.state)
                image_of_pr = image_of_pr[0, pr.equivariant_degree]
                for extra_hom_deg, projs_in_image in image_of_pr.projectives_iter():
                    combined_hom_degree = self.combined_hom_degree(
                        hom_deg, extra_hom_deg
                    )
                    projectives[combined_hom_degree] += projs_in_image

        return MappingProxyType(projectives)

    def _new_differential(self):
        # subdivisions for block structure of matrices
        differential_dict_of_dicts = defaultdict(dict)
        for hom_deg, projs in self.preimage.projectives_iter():
            for a, pr in enumerate(projs):
                image_of_pr = self.derived_functor.on_projectives(pr.state)
                image_of_pr = image_of_pr[0, pr.equivariant_degree]
                for extra_hom_deg, map in image_of_pr.differential:
                    combined_hom_degree = self.combined_hom_degree(
                        hom_deg, extra_hom_deg
                    )
                    differential_dict_of_dicts[combined_hom_degree][a] = map

        differential_degree = self.shift_group().from_homological_part(
            self.embedding_of_new_homological_part(
                self.derived_functor.differential_degree
            )
        )
        differential_hom_degree = differential_degree.homological_part()
        sign = self._new_sign()

        differential = {}
        for (
            combined_hom_degree,
            differential_component_dict,
        ) in differential_dict_of_dicts.items():
            map_dict = {}
            row_subdiv = self._subdivisions[
                combined_hom_degree + differential_hom_degree
            ]
            column_subdiv = self._subdivisions[combined_hom_degree]
            for a, map in differential_component_dict.items():
                row_subdivision = row_subdiv[a]
                column_subdivision = column_subdiv[a]
                # setting a block in the matrix
                for (i, j), entry in map.dict(copy=False).items():
                    map_dict[row_subdivision + i, column_subdivision + j] = entry

            differential[combined_hom_degree] = matrix(
                self.KLRW_algebra().opposite,
                self.component_rank(combined_hom_degree + differential_hom_degree),
                self.component_rank(combined_hom_degree),
                map_dict,
                sparse=True,
                immutable=True,
            )
            differential[combined_hom_degree]._subdivisions = row_subdiv, column_subdiv

        differential = self.DifferentialClass(
            underlying_module=self,
            differential_data=differential,
            degree=differential_degree,
            sign=sign,
        )

        return differential


class ShiftedImageDirectSumOfProjectives(ShiftedKLRWDirectSumOfProjectives):
    @lazy_class_attribute
    def OriginalClass(cls):
        return ImageDirectSumOfProjectives

    def subdivisions(self, grading=None, position=None):
        hom_shift = self.hom_shift()
        if grading is None:
            assert position is None
            return {
                grading - hom_shift: subdivs
                for grading, subdivs in self.original.subdivisions().items()
            }
        grading = self.hom_grading_group()(grading)

        return self.original.subdivisions(grading + hom_shift, position)


class ImagePerfectComplex(ImageDirectSumOfProjectives, KLRWPerfectComplex):
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedImagePerfectComplex

    @lazy_attribute
    def differential(self):
        return self._new_differential()


class ShiftedImagePerfectComplex(
    ShiftedImageDirectSumOfProjectives, ShiftedKLRWPerfectComplex
):
    @lazy_class_attribute
    def OriginalClass(cls):
        return ImagePerfectComplex


class ImagePerfectDoubleComplex(ImageDirectSumOfProjectives, KLRWPerfectMulticomplex):
    @lazy_class_attribute
    def ShiftedClass(cls):
        return ShiftedImagePerfectDoubleComplex

    def _sign_from_old_sign(self, old_sign):
        return self.shift_group().homological_part.sign_from_part(0, old_sign)

    def _degree_from_old_degree(self, old_degree):
        return self.shift_group().summand_embedding(0)(old_degree)

    @lazy_attribute
    def _differentials(self):
        differentials = [self._new_differential()]

        assert isinstance(self.preimage, KLRWPerfectComplex)
        old_differential = self.preimage.differential

        new_diff = self.derived_functor.on_chain_map(old_differential)
        degree = self._degree_from_old_degree(old_differential.degree())
        sign = self._sign_from_old_sign(old_differential._sign)

        ### now check
        complex = KLRWPerfectComplex(
            ring=self.ring(),
            differential=differentials[0],
            differential_degree=differentials[0].degree(),
            sign=differentials[0].sign,
            projectives=self.projectives(),
            extended_grading_group=self.shift_group(),
        )

        diff = complex.hom(complex[degree], new_diff)

        print(">>>>>", (diff * diff).is_zero())
        from klrw.homotopy import homotopy

        print(">>>>>", homotopy(diff * diff, verbose=False) is not None)

        new_diff = self.DifferentialClass(
            underlying_module=self,
            differential_data=new_diff,
            degree=degree,
            sign=sign,
            check=False,  # ???
        )
        differentials += [new_diff]

        # self._check_differentials_(differentials)

        return differentials


class ShiftedImagePerfectDoubleComplex(
    ShiftedImageDirectSumOfProjectives, ShiftedKLRWPerfectMulticomplex
):
    @lazy_class_attribute
    def OriginalClass(cls):
        return ImagePerfectDoubleComplex
"""
