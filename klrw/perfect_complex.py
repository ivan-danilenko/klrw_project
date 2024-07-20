from typing import NamedTuple
from typing import Iterable
from collections import defaultdict
from dataclasses import dataclass, InitVar, replace, field
from types import MappingProxyType
from copy import copy
import operator

from sage.structure.parent import Parent
from sage.groups.additive_abelian.additive_abelian_group import AdditiveAbelianGroup
from sage.groups.additive_abelian.additive_abelian_group import (
    AdditiveAbelianGroupElement,
)
from sage.categories.morphism import Morphism
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.matrix.matrix0 import Matrix
from sage.matrix.constructor import matrix
from sage.modules.free_module import VectorSpace
from sage.structure.element import Vector
from sage.homology.chain_complex import ChainComplex, ChainComplex_class
from sage.rings.integer_ring import ZZ
from sage.rings.finite_rings.integer_mod_ring import IntegerModRing
from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.categories.action import Action

from .klrw_state import KLRWstate
from .klrw_algebra import KLRWAlgebra
from .framed_dynkin import QuiverGradingGroup, QuiverGradingGroupElement


@dataclass(frozen=True)
class KLRWProjectiveModule:
    state: KLRWstate
    equivariant_degree: QuiverGradingGroupElement | int
    grading_group: InitVar[QuiverGradingGroup | None] = None

    def __post_init__(self, grading_group: QuiverGradingGroup | None):
        """
        If grading_group was given coerse grading into an element of it
        """
        if grading_group is not None:
            # bypass protection in frozen=True to change the entry
            super().__setattr__(
                "equivariant_degree",
                grading_group(self.equivariant_degree),
            )

    def __repr__(self):
        return (
            "T_"
            + self.state.__repr__()
            + "{"
            + self.equivariant_degree.__repr__()
            + "}"
        )


@dataclass(frozen=True, eq=True, repr=False)
class KLRWPerfectComplex(Parent):
    """
    Differentials are matrices that act **on the left**.
    Since matrix elements multiply left-to-right,
    we have to use the convention that
    d_i: C_i -> C_{i+degree}
    d_{i-degree}: C_{i-degree} -> C_i
    then the composite map is the matrix multiplication
    d_{i-degree}*d_i.
    This leads to a counterintuitive convention that
    d_i is represented by a matrix of KLRW elements
    that has dim(C_{i+degree}) columns and dim(C_i) rows.
    """

    KLRW_algebra: KLRWAlgebra
    differentials: dict[AdditiveAbelianGroupElement, Matrix] = field(hash=False)
    projectives: (
        dict[AdditiveAbelianGroupElement, Iterable[KLRWProjectiveModule]] | None
    ) = field(hash=False, default=None)
    degree: AdditiveAbelianGroupElement = -1
    grading_group: AdditiveAbelianGroup = ZZ
    mod2grading: Morphism = IntegerModRing(2).coerce_map_from(ZZ)
    check: bool = True
    differentials_hash: int = field(init=False)
    projectives_hash: int = field(init=False)

    def __post_init__(self):
        # normalize projectives and differentials, make immutable
        normalized_projectives = {
            hom_deg: copy(proj) for hom_deg, proj in self.projectives.items() if proj
        }
        normalized_differentials = {
            hom_deg: copy(mat)
            for hom_deg, mat in self.differentials.items()
            if not mat.is_zero()
        }

        for mat in normalized_differentials.values():
            mat.set_immutable()

        normalized_projectives = MappingProxyType(normalized_projectives)
        normalized_differentials

        hashable_differentials = tuple(
            (hom_deg, normalized_differentials[hom_deg])
            for hom_deg in sorted(normalized_differentials.keys())
        )

        hashable_projectives = tuple(
            (key, tuple(normalized_projectives[key]))
            for key in sorted(normalized_projectives.keys())
        )

        # bypass protection in frozen=True to change the entry
        super().__setattr__(
            "differentials_hash",
            hash(hashable_differentials),
        )
        super().__setattr__(
            "projectives_hash",
            hash(hashable_projectives),
        )
        super().__setattr__(
            "differentials",
            MappingProxyType(normalized_differentials),
        )
        super().__setattr__(
            "projectives",
            MappingProxyType(normalized_projectives),
        )

        if self.check:
            for n in self.differentials:
                if n + self.degree in self.differentials:
                    assert (
                        self.differentials[n] * self.differentials[n + self.degree]
                    ).is_zero(), (
                        "d_{} * d_{} != 0".format(n, n + self.degree)
                        + "\n"
                        + repr(
                            self.differentials[n] * self.differentials[n + self.degree]
                        )
                    )

                for (i, j), elem in self.differentials[n].dict(copy=False).items():
                    assert self.projectives[n + self.degree][
                        j
                    ].state == elem.right_state(check_if_all_have_same_right_state=True)
                    assert self.projectives[n][i].state == elem.left_state(
                        check_if_all_have_same_left_state=True
                    )
                    assert (
                        self.projectives[n + self.degree][j].equivariant_degree
                        - self.projectives[n][i].equivariant_degree
                    ) == elem.degree(check_if_homogeneous=True), (
                        repr(self.projectives[n + self.degree][j].equivariant_degree)
                        + " "
                        + repr(self.projectives[n][i].equivariant_degree)
                        + " "
                        + repr(elem.degree(check_if_homogeneous=True))
                    )

    def base_change(self, other_klrw_algebra: KLRWAlgebra):
        new_differentials = {
            hom_deg: diff.change_ring(other_klrw_algebra)
            for hom_deg, diff in self.differentials.items()
        }
        new_projectives = {
            hom_deg: [
                KLRWProjectiveModule(
                    state=other_klrw_algebra.state_set().coerce(proj.state),
                    equivariant_degree=other_klrw_algebra.grading_group.coerce(
                        proj.equivariant_degree
                    ),
                )
                for proj in list_of_projs
            ]
            for hom_deg, list_of_projs in self.projectives.items()
        }

        return self.__class__(
            KLRW_algebra=other_klrw_algebra,
            differentials=new_differentials,
            projectives=new_projectives,
            degree=self.degree,
            grading_group=self.grading_group,
            mod2grading=self.mod2grading,
            check=self.check,
        )

    def rhom_to_simple(
        self,
        simple: KLRWstate,
        dualize_complex=False,
    ) -> ChainComplex_class:
        from sage.groups.additive_abelian.additive_abelian_group import (
            AdditiveAbelianGroup,
        )

        # we grade by ZZ x ZZ
        # these pairs of integers represent
        # (homological_degree, equivariant_degree)
        grading_group = AdditiveAbelianGroup([0, 0])
        degree = grading_group((self.degree, 0))

        relevant_indices = defaultdict(dict)
        for hom_deg in self.projectives:
            for index, proj in enumerate(self.projectives[hom_deg]):
                if proj.state == simple:
                    eq_deg = proj.equivariant_degree
                    grading_pair = (hom_deg, eq_deg.ordinary_grading(as_scalar=True))
                    if grading_pair in relevant_indices:
                        new_index = len(relevant_indices[grading_pair])
                    else:
                        relevant_indices[grading_pair] = {}
                        new_index = 0
                    relevant_indices[grading_pair][index] = new_index

        rhom_diff = {}
        for hom_deg, eq_deg in relevant_indices:
            if dualize_complex:
                codomain_degrees = (hom_deg + self.degree, eq_deg)
            else:
                # since we take RHom(-,I) degree of te differential
                # becomes opposite
                codomain_degrees = (hom_deg - self.degree, eq_deg)

            if codomain_degrees in relevant_indices:
                nrows = len(relevant_indices[codomain_degrees])
            else:
                nrows = 0

            ncols = len(relevant_indices[hom_deg, eq_deg])

            if dualize_complex:
                grading = grading_group((hom_deg, eq_deg))
            else:
                # since we take RHom(-,I) gradings become opposite
                # and domain and codomains swap
                # it gives an extra degree shift
                grading = grading_group((-hom_deg, -eq_deg))

            rhom_diff[grading] = matrix(
                self.KLRW_algebra.scalars(),
                nrows,
                ncols,
            )

        hom = self.KLRW_algebra.base().hom_in_simple()
        for hom_deg, diff in self.differentials.items():
            for (i, j), elem in diff.dict().items():
                left_proj = self.projectives[hom_deg][i]
                if left_proj.state == simple:
                    for braid, coeff in elem:
                        # keep only trivial braids
                        if braid.word() == ():
                            new_elem = hom(coeff)
                            if not new_elem.is_zero():
                                eq_deg = left_proj.equivariant_degree.ordinary_grading(
                                    as_scalar=True
                                )
                                if dualize_complex:
                                    i_new = relevant_indices[
                                        hom_deg + self.degree, eq_deg
                                    ][j]
                                    j_new = relevant_indices[hom_deg, eq_deg][i]

                                    grading = grading_group((hom_deg, eq_deg))
                                else:
                                    # since we take RHom(-,I) gradings become opposite
                                    # and domain and codomains swap
                                    # it gives an extra degree shift
                                    i_new = relevant_indices[hom_deg, eq_deg][i]
                                    j_new = relevant_indices[
                                        hom_deg + self.degree, eq_deg
                                    ][j]

                                    grading = grading_group(
                                        (-hom_deg - self.degree, -eq_deg)
                                    )

                                rhom_diff[grading][i_new, j_new] = new_elem

        return ChainComplex(
            data=rhom_diff,
            degree_of_differential=degree,
            grading_group=grading_group,
        )

    def sign(self, shift):
        shift = self.grading_group(shift)
        if self.mod2grading(shift).is_zero():
            return 1
        return -1

    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            hom_shift, eq_shift = key
        else:
            hom_shift = key
            eq_shift = 0

        sign = self.sign(hom_shift)
        if sign == 1:
            shifted_differentials = {
                grading - hom_shift: diff
                for grading, diff in self.differentials.items()
            }
        elif sign == -1:
            # by default, -matrix is done by using (-1) in the base ring.
            # It's too slow because KLRW has a 1 with maby pieces.
            # Here we do a workaround, by explicitly making
            # the matrix
            shifted_differentials = {}
            for grading, diff in self.differentials.items():
                minus_diff = matrix(
                    self.KLRW_algebra, diff.nrows(), diff.ncols(), sparse=True
                )
                for (i, j), entry in diff.dict(copy=False).items():
                    minus_diff[i, j] = -entry
                shifted_differentials[grading - hom_shift] = minus_diff
        else:
            raise ValueError(sign)

        shifted_projectives = defaultdict(list)
        for grading, projectives_in_grading in self.projectives.items():
            shifted_projectives[grading - hom_shift] = [
                replace(proj, equivariant_degree=proj.equivariant_degree + eq_shift)
                for proj in projectives_in_grading
            ]

        return self.__class__(
            self.KLRW_algebra,
            shifted_differentials,
            shifted_projectives,
            self.degree,
            self.grading_group,
            self.mod2grading,
            check=False,
        )

    def _repr_(self):
        result = "A complex of projectives of a KLRW algebra\n"
        result += "with differentials given by the following matrices\n"
        for grading, diff in self.differentials.items():
            result += "d_{} = \n".format(grading)
            result += repr(diff) + "\n"
        return result


class SummandType(NamedTuple):
    """
    Records combinatorial data about
    Hom(P_i[grading],P_j[grading+shift])
    """

    domain_grading: int  # cohomological grading
    domain_index: int
    codomain_index: int


class KLRWHomOfGradedProjectivesElement(IndexedFreeModuleElement):
    def dict_of_matrices(self):
        result = {}
        for type, coeff in self:
            if type.domain_grading not in result:
                number_of_cols = len(
                    self.parent().codomain.projectives[
                        type.domain_grading + self.parent().shift
                    ]
                )
                number_of_rows = len(
                    self.parent().domain.projectives[type.domain_grading]
                )
                result[type.domain_grading] = matrix(
                    self.parent().KLRW_algebra(),
                    number_of_rows,
                    number_of_cols,
                    sparse=True,
                )
            # similar to convetions for differentials,
            # see docstring for KLRWPerfectComplex
            i = type.domain_index
            j = type.codomain_index
            result[type.domain_grading][i, j] += coeff

        return result

    def cone(self, check=True, keep_subdivisions=True):
        parent = self.parent()
        domain = parent.domain[parent.domain.degree]
        codomain = parent.codomain
        dict_of_matrices = self.dict_of_matrices()

        projectives = defaultdict(list)
        for grading, projs in domain.projectives.items():
            projectives[grading] = copy(projs)
        for grading, projs in codomain.projectives.items():
            projectives[grading].extend(projs)

        differentials = {}
        for grading in projectives:
            next_grading = grading + parent.domain.degree
            if next_grading in projectives:
                differentials[grading] = matrix(
                    codomain.KLRW_algebra,
                    len(projectives[grading]),
                    len(projectives[next_grading]),
                    sparse=True,
                )
                # the matrix has block structure
                # rows are splitted into two categories by top_block_size
                if grading in domain.projectives:
                    top_block_size = len(domain.projectives[grading])
                else:
                    top_block_size = 0
                # columns are splitted into two categories by left_block_size
                if next_grading in domain.projectives:
                    left_block_size = len(domain.projectives[next_grading])
                else:
                    left_block_size = 0

                if grading in domain.differentials:
                    differentials[grading].set_block(
                        0, 0, domain.differentials[grading]
                    )
                if next_grading in dict_of_matrices:
                    differentials[grading].set_block(
                        0, left_block_size, dict_of_matrices[next_grading]
                    )
                if grading in codomain.differentials:
                    differentials[grading].set_block(
                        top_block_size, left_block_size, codomain.differentials[grading]
                    )
                if keep_subdivisions:
                    differentials[grading]._subdivisions = (
                        [0, top_block_size, len(projectives[grading])],
                        [0, left_block_size, len(projectives[next_grading])],
                    )

        cone = KLRWPerfectComplex(
            codomain.KLRW_algebra,
            differentials=differentials,
            projectives=projectives,
            check=check,
        )

        return cone

    def to_dict_of_CSR(self, transpose=True):
        """
        Transforms into a dictionary of CSR matrices.

        Beware of the transposition in the convention!
        [see the comment in KLRWPerfectComplex]
        By default, transpose=True, it follows this convention
        """
        from klrw.cython_exts.sparse_csr import CSR_Mat

        self_dict = defaultdict(dict)
        for stype, coeff in self:
            hom_deg, i, j = stype
            if transpose:
                self_dict[hom_deg][i, j] = coeff
            else:
                self_dict[hom_deg][j, i] = coeff

        number_of_columns = {
            hom_deg: len(
                self.parent().codomain.projectives[hom_deg + self.parent().shift]
            )
            for hom_deg in self_dict
        }
        number_of_rows = {
            hom_deg: len(self.parent().domain.projectives[hom_deg])
            for hom_deg in self_dict
        }
        if not transpose:
            number_of_columns, number_of_rows = number_of_rows, number_of_columns

        return {
            hom_deg: CSR_Mat.from_dict(
                self_dict[hom_deg],
                number_of_rows[hom_deg],
                number_of_columns[hom_deg],
            )
            for hom_deg in self_dict
        }

    def to_dict_of_CSC(self, transpose=True):
        """
        Transforms into a dictionary of CSC matrices.

        Beware of the transposition in the convention!
        [see the comment in KLRWPerfectComplex]
        By default, transpose=True, it follows this convention
        """
        from klrw.cython_exts.sparse_csc import CSC_Mat

        self_dict = defaultdict(dict)
        for stype, coeff in self:
            hom_deg, i, j = stype
            if transpose:
                self_dict[hom_deg][i, j] = coeff
            else:
                self_dict[hom_deg][j, i] = coeff

        number_of_columns = {
            hom_deg: len(
                self.parent().codomain.projectives[hom_deg + self.parent().shift]
            )
            for hom_deg in self_dict
        }
        number_of_rows = {
            hom_deg: len(self.parent().domain.projectives[hom_deg])
            for hom_deg in self_dict
        }
        if not transpose:
            number_of_columns, number_of_rows = number_of_rows, number_of_columns

        return {
            hom_deg: CSC_Mat.from_dict(
                self_dict[hom_deg],
                number_of_rows[hom_deg],
                number_of_columns[hom_deg],
            )
            for hom_deg in self_dict
        }

    def _mul_(self, other):
        """
        Used *only* in the special case when both self and other
        have the same parent.
        Because of a technical details in the coercion model
        this method is called instead of action.
        """
        return self.parent().self_action.act(self, other)

    def _repr_(self):
        dict_of_matrices = self.dict_of_matrices().items()
        if dict_of_matrices:
            result = ""
            for grading, mat in self.dict_of_matrices().items():
                result += (
                    grading.__repr__()
                    + " -> "
                    + (grading + self.parent().shift).__repr__()
                    + ":\n"
                )
                result += mat.__repr__() + "\n"
            return result
        else:
            return "0"


class HomHomMultiplication(Action):
    """
    Gives multiplication of two homs of graded projectives.

    The solution is similar to matrix multiplication is Sage.
    [see MatrixMatrixAction in MatrixSpace]
    Coercion model in Sage requires both elements in
    the usual product to have the same parent, but in
    actions this is no longer the case. So we use the class Action.
    """

    def __init__(self, left_parent, right_parent):
        Action.__init__(
            self, G=left_parent, S=right_parent, is_left=True, op=operator.mul
        )
        self._left_domain = self.actor()
        self._right_domain = self.domain()
        self._codomain = self._create_codomain()

    def _create_codomain(self):
        if (
            self._left_domain.domain.projectives
            != self._right_domain.codomain.projectives
        ):
            from pprint import pprint

            pprint(self._left_domain.domain.projectives)
            pprint(self._right_domain.codomain.projectives)
            raise TypeError("Incomposable homs")
        return KLRWHomOfGradedProjectives(
            domain=self._right_domain.domain,
            codomain=self._left_domain.codomain,
            shift=self._left_domain.shift + self._right_domain.shift,
        )

    def _act_(
        self,
        g: KLRWHomOfGradedProjectivesElement,
        s: KLRWHomOfGradedProjectivesElement,
    ) -> KLRWHomOfGradedProjectivesElement:
        from klrw.cython_exts.sparse_multiplication import multiplication

        left_dict = g.to_dict_of_CSC()
        right_dict = s.to_dict_of_CSR()
        result_dict = {}
        for hom_deg in left_dict:
            shifted_hom_deg = hom_deg + self._right_domain.shift
            if shifted_hom_deg in right_dict:
                # because of the transposition convention
                # the first map has to go on the left in the matrix
                # multiplication but in the composition it goes on
                # the right
                result_dict[hom_deg] = multiplication(
                    left=right_dict[hom_deg],
                    right=left_dict[shifted_hom_deg],
                    as_dict=True,
                )

        return self._codomain._element_from_dict_of_dicts_(result_dict, transpose=True)


class ParameterHomMultiplication(Action):
    """
    Multiplication of a hom of graded projectives
    by a symmetric homogeneous element in the dot algebra.

    Default coercion uses the unit in KLRW which is not the best
    way to do multiplication by parameters.
    """

    def __init__(self, other, hom_parent):
        self.dot_algebra = hom_parent.KLRW_algebra().base()
        self.coerce_map = self.dot_algebra.coerce_map_from(other)
        Action.__init__(self, G=other, S=hom_parent, is_left=True, op=operator.mul)

    def codomain(self):
        raise AttributeError("Codomain depend on the dot algebra element")

    def _act_(
        self,
        g,
        h: KLRWHomOfGradedProjectivesElement,
    ) -> KLRWHomOfGradedProjectivesElement:
        g = self.coerce_map(g)
        grading_group = self.domain().KLRW_algebra().grading_group
        element_degree = self.dot_algebra.element_degree(
            g,
            grading_group,
            check_if_homogeneous=True,
        )
        assert self.dot_algebra.is_element_symmetric(g), "The element is not symmetric"

        hom_codomain_projectives = {}
        h_codomain = self.domain().codomain
        for hom_deg, projs in h_codomain.projectives.items():
            hom_codomain_projectives[hom_deg] = [
                replace(pr, equivariant_degree=pr.equivariant_degree + element_degree)
                for pr in projs
            ]

        hom_codomain = KLRWPerfectComplex(
            KLRW_algebra=self.domain().codomain.KLRW_algebra,
            differentials=self.domain().codomain.differentials,
            projectives=hom_codomain_projectives,
            degree=self.domain().codomain.degree,
            grading_group=self.domain().codomain.grading_group,
            mod2grading=self.domain().codomain.mod2grading,
            check=True,
        )

        product_parent = KLRWHomOfGradedProjectives(
            self.domain().domain,
            hom_codomain,
            shift=self.domain().shift,
        )

        elem_times_h_dict = {stype: g * elem for stype, elem in h}

        return product_parent._from_dict(elem_times_h_dict)


class KLRWHomOfGradedProjectives(CombinatorialFreeModule):
    @staticmethod
    def __classcall__(
        cls,
        domain: KLRWPerfectComplex,
        codomain: KLRWPerfectComplex,
        shift=0,
    ):
        return super().__classcall__(
            cls,
            domain=domain,
            codomain=codomain,
            shift=shift,
        )

    def __init__(
        self,
        domain: KLRWPerfectComplex,
        codomain: KLRWPerfectComplex,
        shift,
    ):
        assert domain.KLRW_algebra is codomain.KLRW_algebra
        self.domain = domain
        self.codomain = codomain
        self.shift = shift
        super().__init__(
            R=self.domain.KLRW_algebra,
            basis_keys=self._types_(),
            element_class=KLRWHomOfGradedProjectivesElement,
        )

    def KLRW_algebra(self):
        return self.domain.KLRW_algebra

    @cached_method
    def _types_(self):
        return tuple(self._types_iter_())

    def types(self, index=None):
        if index is not None:
            return self._types_()[index]
        return self._types_()

    def _types_iter_(self):
        for grading in self.domain.projectives:
            for i in range(len(self.domain.projectives[grading])):
                if grading + self.shift in self.codomain.projectives:
                    for j in range(
                        len(self.codomain.projectives[grading + self.shift])
                    ):
                        type = SummandType(
                            domain_grading=grading,
                            domain_index=i,
                            codomain_index=j,
                        )
                        dim = self.graded_component_of_type(type).dimension()
                        if dim > 0:
                            yield type

    @cached_method
    def _subdivisions_(self):
        """
        The vector over base ring encoding of the elements has block structure.
        Here we record the indices where the borders between the blocks are.
        If index is given, returns _subdivisions_()[index]
        """
        subdivisions = [0]
        for t in self.types():
            dim = self.graded_component_of_type(t).dimension()
            subdivisions.append(subdivisions[-1] + dim)
        return tuple(subdivisions)

    def subdivisions(self, index=None):
        """
        The vector over base ring encoding of the elements has block structure.
        Here we record the indices where the borders between the blocks are.
        If index is given, returns _subdivisions_()[index]
        """
        if index is not None:
            return self._subdivisions_()[index]
        return self._subdivisions_()

    def dimension(self):
        return self.subdivisions(-1)

    def vector_space(self):
        return VectorSpace(self.KLRW_algebra().scalars(), self.dimension(), sparse=True)

    def graded_component_of_type(self, type):
        left = self.domain.projectives[type.domain_grading][type.domain_index]
        codomain_grading = type.domain_grading + self.shift
        right = self.codomain.projectives[codomain_grading][type.codomain_index]
        left_state = left.state
        right_state = right.state
        degree = right.equivariant_degree - left.equivariant_degree
        return self.KLRW_algebra()[left_state:right_state:degree]

    @lazy_attribute
    def _type_to_subdivision_index_(self) -> MappingProxyType:
        """
        For a type gives the index where the corresponding subdivision starts.
        """
        result = {}
        for i, type in enumerate(self.types()):
            result[type] = self.subdivisions(i)
        return MappingProxyType(result)

    def _element_from_vector_(self, vect):
        assert len(vect) == self.subdivisions(-1)

        result_dict = {}
        for ind in range(len(self.subdivisions()) - 1):
            type = self.types()[ind]

            begin = self.subdivisions(ind)
            end = self.subdivisions(ind + 1)
            graded_component = self.graded_component_of_type(type)
            entry = graded_component._element_from_vector_(vect[begin:end])

            if not entry.is_zero():
                result_dict[type] = entry

        return self._from_dict(result_dict)

    def _vector_from_element_(self, elem):
        assert elem.parent() is self
        # create a mutable zero vector
        vect = self.vector_space()()
        for type, coeff in elem:
            graded_component = self.graded_component_of_type(type)
            begin = self._type_to_subdivision_index_[type]
            vector_part = graded_component._vector_from_element_(coeff)
            for rel_ind, scalar in vector_part.dict(copy=False).items():
                vect[begin + rel_ind] = scalar

        return vect

    def _element_from_dict_of_dicts_(self, dic: dict, transpose=True):
        """
        Transforms a dictionary of dictionaries into a hom element.

        Beware of the transposition in the convention!
        [see the comment in KLRWPerfectComplex]
        By default, transpose=True, it follows this convention
        """
        if transpose:
            result_dict = {
                SummandType(
                    domain_grading=hom_deg,
                    domain_index=i,
                    codomain_index=j,
                ): entry
                for hom_deg, mat in dic.items()
                for (i, j), entry in mat.items()
            }
        else:
            result_dict = {
                SummandType(
                    domain_grading=hom_deg,
                    domain_index=j,
                    codomain_index=i,
                ): entry
                for hom_deg, mat in dic.items()
                for (i, j), entry in mat.items()
            }
        return self._from_dict(result_dict)

    def _get_action_(self, other, op, self_on_left):
        if op == operator.mul:
            if isinstance(other, KLRWHomOfGradedProjectives):
                if self_on_left:
                    return HomHomMultiplication(left_parent=self, right_parent=other)
                else:
                    return HomHomMultiplication(left_parent=other, right_parent=self)
            if self.KLRW_algebra().base().has_coerce_map_from(other):
                return ParameterHomMultiplication(other, self)
        return super()._get_action_(other, op, self_on_left)

    @lazy_attribute
    def self_action(self):
        return HomHomMultiplication(left_parent=self, right_parent=self)

    def _repr_(self):
        return "Graded morphims between graded modules"

    def one(self, as_vector=False):
        assert self.shift == 0
        assert self.domain.projectives == self.codomain.projectives

        one = self.zero()
        for hom_deg, projs in self.domain.projectives.items():
            for index, pr in enumerate(projs):
                type = SummandType(
                    domain_grading=hom_deg,
                    domain_index=index,
                    codomain_index=index,
                )
                one += self.term(type, coeff=self.KLRW_algebra().idempotent(pr.state))

        if as_vector:
            one = self._vector_from_element_(one)

        return one


class KLRWExtOfGradedProjectives(Parent):
    def __init__(self, ambient, shift):
        self._shift = shift
        self._ambient = ambient

    @lazy_attribute
    def _cycle_module_(self):
        current_differential_matrix = self._ambient._differential_matrix_(self._shift)

        return current_differential_matrix.right_kernel()

    @lazy_attribute
    def _boundary_module_(self):
        previous_differential_matrix = self._ambient._differential_matrix_(
            self._shift - self._ambient.degree()
        )

        return previous_differential_matrix.column_module()

    @lazy_attribute
    def _homology_(self):
        cyc = self._cycle_module_
        bon = self._boundary_module_
        quo = cyc.quotient(bon)
        return quo

    @lazy_attribute
    def _basis_vectors_(self):
        return tuple(self._homology_.lift(b) for b in self._homology_.basis())

    @lazy_attribute
    def _relations_vectors_(self):
        return tuple(b for b in self._boundary_module_.basis())

    @lazy_attribute
    def _vector_space_mod_boundaries_(self):
        hom = self.hom_of_graded
        return hom.vector_space() / self._boundary_module_

    @lazy_attribute
    def _vector_quotient_map_(self):
        return self._vector_space_mod_boundaries_.quotient_map()

    @lazy_attribute
    def _vector_lift_map_(self):
        return self._vector_space_mod_boundaries_.lift_map()

    @lazy_attribute
    def hom_of_graded(self):
        return self._ambient[self._shift]

    @cached_method
    def basis(self, as_vectors=False):
        if as_vectors:
            return self._basis_vectors_
        else:
            hom = self.hom_of_graded
            return tuple(hom._element_from_vector_(b) for b in self._basis_vectors_)

    def dimension(self):
        return len(self._basis_vectors_)
    
    def relations(self, as_vectors=False):
        if as_vectors:
            return self._relations_vectors_
        else:
            hom = self.hom_of_graded
            return tuple(hom._element_from_vector_(b) for b in self._relations_vectors_)

    def KLRW_algebra(self):
        return self._ambient.KLRW_algebra()

    def reduce(self, element):
        if isinstance(element, KLRWHomOfGradedProjectivesElement):
            hom = self.hom_of_graded
            vect = hom._vector_from_element_(element)
            vect = self.reduce(vect)
            return hom._element_from_vector_(vect)
        elif isinstance(element, Vector):
            quotient_map = self._vector_quotient_map_
            lift_map = self._vector_lift_map_
            return lift_map(quotient_map(element))
        else:
            raise NotImplementedError

    def if_homotopic_to_scalar(self, element):
        if isinstance(element, KLRWHomOfGradedProjectivesElement):
            hom = self.hom_of_graded
            vect = hom._vector_from_element_(element)
            vect = self.reduce(vect)
            return self.if_homotopic_to_scalar(vect)
        elif isinstance(element, Vector):
            hom = self.hom_of_graded
            one_reduced = self.reduce(hom.one(as_vector=True))
            elem_reduced = self.reduce(element)
            return (
                span(
                    [one_reduced, elem_reduced], self.KLRW_algebra().scalars()
                ).dimension()
                <= 1
            )
        else:
            raise NotImplementedError


class KLRWHomOfPerfectComplexes(Parent):
    def __init__(
        self,
        domain: KLRWPerfectComplex,
        codomain: KLRWPerfectComplex,
    ):
        assert domain.KLRW_algebra == codomain.KLRW_algebra
        assert domain.degree == codomain.degree
        assert domain.grading_group is codomain.grading_group
        assert domain.mod2grading == codomain.mod2grading
        self.domain = domain
        self.codomain = codomain

    def KLRW_algebra(self):
        return self.domain.KLRW_algebra

    def degree(self):
        return self.domain.degree

    def sign(self, shift):
        return self.domain.sign(shift)

    def _differential_matrix_(self, shift, keep_subdivisions=True):
        """
        Makes a matrix of the differential that acts on graded morphisms
        domain->codomain[shift]
        The basis is the one in
        KLRWHomOfGradedProjectives(self.domain, self.codomain, ...)
        """
        current_hom = KLRWHomOfGradedProjectives(self.domain, self.codomain, shift)
        next_hom = KLRWHomOfGradedProjectives(
            self.domain, self.codomain, shift + self.degree()
        )

        diff_mat = matrix(
            self.KLRW_algebra().scalars(),
            next_hom.dimension(),
            current_hom.dimension(),
            sparse=True,
        )
        if keep_subdivisions:
            diff_mat._subdivisions = (
                list(next_hom.subdivisions()),
                list(current_hom.subdivisions()),
            )

        # matrix of left multiplication by d.
        for type in current_hom.types():
            column_subdivision_index = current_hom._type_to_subdivision_index_[type]
            domain_grading = type.domain_grading
            if domain_grading + shift in self.codomain.differentials:
                klrw_mat = self.codomain.differentials[domain_grading + shift]
                for (i, j), d_entry in klrw_mat.dict(copy=False).items():
                    if i == type.codomain_index:
                        product_type = SummandType(
                            domain_grading=type.domain_grading,
                            domain_index=type.domain_index,
                            codomain_index=j,
                        )
                        # product type may be missing if the dimension of the
                        # corresponding graded component is zero.
                        if product_type in next_hom._type_to_subdivision_index_:
                            row_subdivision_index = (
                                next_hom._type_to_subdivision_index_[product_type]
                            )
                            graded_component = current_hom.graded_component_of_type(
                                type
                            )
                            # acting_on_left=False is because
                            # [despite d acts on the left]
                            # matrix coefficients multiply from the right
                            # see the docstring in KLRWPerfectComplex
                            submatrix = d_entry.as_matrix_in_graded_component(
                                graded_component, acting_on_left=False
                            )

                            diff_mat.set_block(
                                row_subdivision_index,
                                column_subdivision_index,
                                submatrix,
                            )

        # matrix of right multiplication by d.
        # there is a sign -(-1)^degree, it is -self.sign(shift)
        for type in current_hom.types():
            column_subdivision_index = current_hom._type_to_subdivision_index_[type]
            domain_grading = type.domain_grading - self.degree()
            if domain_grading in self.domain.differentials:
                klrw_mat = self.domain.differentials[domain_grading]
                for (i, j), d_entry in klrw_mat.dict(copy=False).items():
                    if j == type.domain_index:
                        product_type = SummandType(
                            domain_grading=domain_grading,
                            domain_index=i,
                            codomain_index=type.codomain_index,
                        )
                        # product type may be missing if the dimension of the
                        # corresponding graded component is zero.
                        if product_type in next_hom._type_to_subdivision_index_:
                            row_subdivision_index = (
                                next_hom._type_to_subdivision_index_[product_type]
                            )
                            graded_component = current_hom.graded_component_of_type(
                                type
                            )
                            # acting_on_left=True is because
                            # [despite d acts on the right]
                            # matrix coefficients multiply from the right
                            # see the docstring in KLRWPerfectComplex
                            submatrix = d_entry.as_matrix_in_graded_component(
                                graded_component, acting_on_left=True
                            )
                            sign = -self.sign(shift)
                            if sign != 1:
                                submatrix *= sign

                            for (a, b), scalar in submatrix.dict(copy=False).items():
                                diff_mat[
                                    row_subdivision_index + a,
                                    column_subdivision_index + b,
                                ] += scalar

        return diff_mat

    @cached_method
    def __getitem__(self, shift):
        """
        Returns maps of graded spaces
        """
        return KLRWHomOfGradedProjectives(self.domain, self.codomain, shift)

    def ext(self, shift):
        """
        Returns Ext^shift
        """
        # ???? MAKE SHIFTS
        ext = KLRWExtOfGradedProjectives(ambient=self, shift=shift)

        return ext

    def one(self, as_vector=False):
        return self[0].one(as_vector=as_vector)
