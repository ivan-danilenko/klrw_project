import cython
import numpy as np
from itertools import count
import bisect

from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.ring import PrincipalIdealDomain
from sage.matrix.matrix_sparse import Matrix_sparse
from sage.matrix.constructor import matrix
from sage.combinat.root_system.cartan_type import CartanType

from .klrw_algebra import KLRWAlgebra
from .framed_dynkin import FramedDynkinDiagram_with_dimensions
from .cython_exts.solver import Solver
from .cython_exts.sparse_csc import CSC_Mat


class TThimble:
    __slots__ = ("segment", "hom_deg", "equ_deg", "order")

    def __init__(self, segment, hom_deg, equ_deg, order):
        self.segment = segment
        self.hom_deg = hom_deg
        self.equ_deg = equ_deg
        self.order = order

    def __repr__(self):
        return (
            "T-Thimble in "
            + self.segment.__repr__()
            + " segment with "
            + self.hom_deg.__repr__()
            + " cohomological degree and "
            + self.equ_deg.__repr__()
            + " equivariant degree, on "
            + self.order.__repr__()
            + " position"
        )


class ProductThimbles:
    """
    colored_state is a list or tuple where i-th element is the position of the i-th
    moving strand
    frozenset(...) of this list gives the (uncolored) state in KLRW elements
    order is a tuple of how many intersection points are to the left for each
    intersection
    intersection_indices is a tuple to remember the indices of thimbles and access
    information about them
    """

    __slots__ = (
        "colored_state",
        "hom_deg",
        "equ_deg",
        "order",
        "next_thimble_strand_number",
        "next_thimble_position",
        "intersection_indices",
    )

    def __init__(
        self,
        colored_state: list | tuple,
        hom_deg,
        equ_deg,
        order: list | tuple,
        next_thimble_strand_number,
        next_thimble_position,
        intersection_indices,
    ):
        self.colored_state = colored_state
        self.hom_deg = hom_deg
        self.equ_deg = equ_deg
        self.order = order
        self.next_thimble_strand_number = next_thimble_strand_number
        self.next_thimble_position = next_thimble_position
        self.intersection_indices = intersection_indices

    def __repr__(self):
        return (
            "Product of T-Thimbles: "
            + self.colored_state.__repr__()
            + ", "
            + self.hom_deg.__repr__()
            + ", "
            + self.equ_deg.__repr__()
            + ", "
            + self.order.__repr__()
            + ", "
            + self.equ_deg.__repr__()
            + ", "
            + self.next_thimble_strand_number.__repr__()
            + ", "
            + self.next_thimble_position.__repr__()
            + ", "
            + self.intersection_indices.__repr__()
        )

    def uncolored_state(self):
        return frozenset(self.colored_state)


class CombinatorialEBrane:
    def __init__(self, number_of_punctures, number_of_E_branes):
        self.n = number_of_punctures
        self.k = number_of_E_branes

        self.quiver = FramedDynkinDiagram_with_dimensions(CartanType(["A", 1]))
        self.V, self.W = self.quiver.vertices()
        self.quiver[self.W] = self.n

        # we will use several KLRW algebras
        # one for each of the subset of the moving strands
        self.klrw_algebra = {}
        for i in range(1, self.k + 1):
            self.quiver[self.V] = i
            self.klrw_algebra[i] = KLRWAlgebra(
                ZZ,
                self.quiver,
                vertex_prefix="h",
                framing_prefix="u",
                warnings=True,
            )
            self.klrw_algebra[i].braid_set().enable_checks()
            print("****")

        # We need to convert elements of one algebra of dots
        # to elements of the one with more strands.
        # self.hom_one_to_many_dots[i,j] is the homomorphism
        # from the algebra of dots on one strand to the algebra
        # on i strands that sends all dots to dots on j-th strand.
        self.hom_one_to_many_dots = {}
        domain_dots_algebra = self.klrw_algebra[1].base()
        for i in range(1, self.k):
            codomain_dots_algebra = self.klrw_algebra[i + 1].base()
            for j in range(1, i + 2):
                # dots on the only strand go to dots on j-th strand
                map = {(self.V, 1): (self.V, j)}
                hom = domain_dots_algebra.hom_from_dots_map(codomain_dots_algebra, map)
                self.hom_one_to_many_dots[i, j] = hom

        # self.hom_add_one_more_strand[i,j] is the homomorphism
        # from the algebra of dots on i strands to the algebra
        # on i+1 strands keeping j-th strand without dots.
        self.hom_add_one_more_strand = {}
        for i in range(1, self.k):
            domain_dots_algebra = self.klrw_algebra[i].base()
            codomain_dots_algebra = self.klrw_algebra[i + 1].base()
            for j in range(1, i + 2):
                # dots on the only strand go to dots on j-th strand
                map = {
                    (self.V, k): (self.V, k) if k < j else (self.V, k + 1)
                    for k in range(1, i + 1)
                }
                hom = domain_dots_algebra.hom_from_dots_map(codomain_dots_algebra, map)
                self.hom_add_one_more_strand[i, j] = hom

        E_brane_cyclic = [0, 1, 2, 3]
        E_brane_intersections = [[0], [1, 3], [2]]
        # Warning about equivariant grading: we picked the convention
        # from https://arxiv.org/pdf/0804.2080.pdf
        # The convention from https://arxiv.org/pdf/2305.13480.pdf
        # is asymmetric, i.e. depends on
        # if we read the braid right-to-lefttom or lefttom-to-right
        E_branes_intersections_data = {
            0: TThimble(segment=0, hom_deg=0, equ_deg=0, order=None),
            1: TThimble(segment=1, hom_deg=-1, equ_deg=1, order=None),
            2: TThimble(segment=2, hom_deg=0, equ_deg=0, order=None),
            3: TThimble(segment=1, hom_deg=1, equ_deg=-1, order=None),
        }
        E_brane_length = len(E_brane_intersections)
        intersection_points_in_E_brane = sum(len(lis) for lis in E_brane_intersections)
        # Here we keep the intesection points separated into bins between the punctures
        self.intersections = [list() for i in range(number_of_punctures + 1)]
        self.intersections_data = {}

        assert number_of_E_branes * (E_brane_length - 1) <= number_of_punctures

        self.branes = [
            [j + i * len(E_brane_cyclic) for j in E_brane_cyclic]
            for i in range(number_of_E_branes)
        ]

        for copy_number in range(number_of_E_branes):
            for j in range(E_brane_length):
                for i in E_brane_intersections[j]:
                    new_point = i + copy_number * intersection_points_in_E_brane
                    segment = j + copy_number * (E_brane_length - 1)
                    hom_degree = E_branes_intersections_data[i].hom_deg
                    equ_degree = E_branes_intersections_data[i].equ_deg
                    self.intersections[segment].append(new_point)
                    self.intersections_data[new_point] = TThimble(
                        segment=segment,
                        hom_deg=hom_degree,
                        equ_deg=equ_degree,
                        order=None,
                    )

        # keeps track of branch in log(yi-yj)
        # initially everything in the principal branch
        self.pairwise_equ_deg = {}
        for b1 in range(len(self.branes)):
            for b0 in range(b1):
                for pt0 in self.branes[b0]:
                    for pt1 in self.branes[b1]:
                        self.pairwise_equ_deg[pt0, pt1] = 0

        self.number_of_intersections = (
            number_of_E_branes * intersection_points_in_E_brane
        )

        self.differential = None
        self.thimbles = None

    def apply_s(self, i):
        assert i != 0
        if i > 0:
            sign = +1
        else:
            sign = -1
            i = -i
        assert i < len(self.intersections)

        new_points_left = []
        new_points_right = []
        for point in self.intersections[i]:
            # we add two new points, one in the segment on the left,
            # the other in the segment on the right
            left_point = self.number_of_intersections
            right_point = self.number_of_intersections + 1
            new_points_left.append(left_point)
            new_points_right.append(right_point)
            self.number_of_intersections += 2

            hom_deg = self.intersections_data[point].hom_deg
            equ_deg = self.intersections_data[point].equ_deg
            if sign == +1:
                self.intersections_data[left_point] = TThimble(
                    segment=i - 1, hom_deg=hom_deg, equ_deg=equ_deg - 1, order=None
                )
                self.intersections_data[right_point] = TThimble(
                    segment=i + 1, hom_deg=hom_deg, equ_deg=equ_deg - 1, order=None
                )
                self.intersections_data[point] = TThimble(
                    segment=i, hom_deg=hom_deg + 1, equ_deg=equ_deg - 2, order=None
                )
            else:
                self.intersections_data[left_point] = TThimble(
                    segment=i - 1, hom_deg=hom_deg, equ_deg=equ_deg + 1, order=None
                )
                self.intersections_data[right_point] = TThimble(
                    segment=i + 1, hom_deg=hom_deg, equ_deg=equ_deg + 1, order=None
                )
                self.intersections_data[point] = TThimble(
                    segment=i, hom_deg=hom_deg - 1, equ_deg=equ_deg + 2, order=None
                )

            # now modify the parts on the branes
            # using "b in range(...)" instead of "b in self.branes"
            # because we need to modify elements of self.branes
            for b in range(len(self.branes)):
                if point in self.branes[b]:
                    ind = self.branes[b].index(point)
                    # if odd, so goes "down" before cohomological shift
                    if ind % 2:
                        if sign == +1:
                            self.branes[b] = (
                                self.branes[b][:ind]
                                + [left_point, point, right_point]
                                + self.branes[b][ind + 1 :]
                            )
                        else:
                            self.branes[b] = (
                                self.branes[b][:ind]
                                + [right_point, point, left_point]
                                + self.branes[b][ind + 1 :]
                            )
                    # if even, so goes "up" before cohomological shift
                    else:
                        if sign == +1:
                            self.branes[b] = (
                                self.branes[b][:ind]
                                + [right_point, point, left_point]
                                + self.branes[b][ind + 1 :]
                            )
                        else:
                            self.branes[b] = (
                                self.branes[b][:ind]
                                + [left_point, point, right_point]
                                + self.branes[b][ind + 1 :]
                            )

        # now modify the pairwise contributions to the equivariant degree
        # we do it only on this step because of the case when lefth points
        # are on i-th segement
        old_keys = [x for x in self.pairwise_equ_deg]
        for pts in old_keys:
            # pts is a 2-tuple
            if_pt_on_the_segment = [p in self.intersections[i] for p in pts]
            # if lefth points are on the segment
            if if_pt_on_the_segment[0] and if_pt_on_the_segment[1]:
                # find the places of pt0,pt1 and corresponding left&right points
                js = [self.intersections[i].index(p) for p in pts]

                left_points = [new_points_left[j] for j in js]
                right_points = [new_points_right[j] for j in js]

                self.pairwise_equ_deg[left_points[0], left_points[1]] = (
                    self.pairwise_equ_deg[pts]
                )
                self.pairwise_equ_deg[right_points[0], right_points[1]] = (
                    self.pairwise_equ_deg[pts]
                )
                # for many cases the value depents on relative positions
                # of points on the segment
                if js[0] < js[1]:
                    self.pairwise_equ_deg[left_points[0], pts[1]] = (
                        self.pairwise_equ_deg[pts]
                    )
                    self.pairwise_equ_deg[left_points[0], right_points[1]] = (
                        self.pairwise_equ_deg[pts]
                    )
                    self.pairwise_equ_deg[pts[0], left_points[1]] = (
                        self.pairwise_equ_deg[pts] + 2 * sign
                    )
                    self.pairwise_equ_deg[pts[0], right_points[1]] = (
                        self.pairwise_equ_deg[pts]
                    )
                    self.pairwise_equ_deg[right_points[0], left_points[1]] = (
                        self.pairwise_equ_deg[pts] + 2 * sign
                    )
                    self.pairwise_equ_deg[right_points[0], pts[1]] = (
                        self.pairwise_equ_deg[pts] + 2 * sign
                    )
                else:
                    self.pairwise_equ_deg[left_points[0], pts[1]] = (
                        self.pairwise_equ_deg[pts] + 2 * sign
                    )
                    self.pairwise_equ_deg[left_points[0], right_points[1]] = (
                        self.pairwise_equ_deg[pts] + 2 * sign
                    )
                    self.pairwise_equ_deg[pts[0], left_points[1]] = (
                        self.pairwise_equ_deg[pts]
                    )
                    self.pairwise_equ_deg[pts[0], right_points[1]] = (
                        self.pairwise_equ_deg[pts] + 2 * sign
                    )
                    self.pairwise_equ_deg[right_points[0], left_points[1]] = (
                        self.pairwise_equ_deg[pts]
                    )
                    self.pairwise_equ_deg[right_points[0], pts[1]] = (
                        self.pairwise_equ_deg[pts]
                    )
                # we change this the last because it's used in all the other cases
                self.pairwise_equ_deg[pts] += 2 * sign

            elif if_pt_on_the_segment[0]:
                # and p1 is not on the segment from if statement above
                # find the place of pt0 and corresponding left&right points
                j = self.intersections[i].index(pts[0])
                left_point = new_points_left[j]
                right_point = new_points_right[j]
                # keep self.pairwise_equ_deg[pt0,pt1] the same
                self.pairwise_equ_deg[left_point, pts[1]] = self.pairwise_equ_deg[pts]
                self.pairwise_equ_deg[right_point, pts[1]] = self.pairwise_equ_deg[pts]

            elif if_pt_on_the_segment[1]:
                # and p1 is not on the segment from if statement above
                # find the place of pt0 and corresponding left&right points
                j = self.intersections[i].index(pts[1])
                left_point = new_points_left[j]
                right_point = new_points_right[j]
                # keep self.pairwise_equ_deg[pt0,pt1] the same
                self.pairwise_equ_deg[pts[0], left_point] = self.pairwise_equ_deg[pts]
                self.pairwise_equ_deg[pts[0], right_point] = self.pairwise_equ_deg[pts]

        self.intersections[i - 1] = self.intersections[i - 1] + new_points_left
        self.intersections[i + 1] = new_points_right + self.intersections[i + 1]
        self.intersections[i] = self.intersections[i][::-1]

        self.simplify_branes()

    def simplify_branes(self):
        # using "b in range(...)" instead of "b in self.branes"
        # because we need to modify elements of self.branes
        for b in range(len(self.branes)):
            current_index = 0
            # use this instead of for because we might need to
            # decrease current_index in the process
            # and change the length of self.branes[b]
            while current_index != len(self.branes[b]):
                # if not the end
                if current_index != len(self.branes[b]) - 1:
                    next_index = current_index + 1
                    current_point = self.branes[b][current_index]
                    next_point = self.branes[b][next_index]

                    if (
                        self.intersections_data[current_point].segment
                        == self.intersections_data[next_point].segment
                    ):
                        self.branes[b] = (
                            self.branes[b][:current_index]
                            + self.branes[b][next_index + 1 :]
                        )
                        # if cancellation happened, there is a chance
                        # the previous point can be cancelled now
                        current_index -= 1
                    else:
                        current_index += 1

                # take cyclicity into account
                else:
                    current_point = self.branes[b][current_index]
                    next_point = self.branes[b][0]
                    if (
                        self.intersections_data[current_point].segment
                        == self.intersections_data[next_point].segment
                    ):
                        # delete the first and the last element
                        # we also need to keep even positions even and odd ones odd
                        # to make sure going upwards and downwards are consistent
                        # so we move self.branes[b][1] to the end
                        self.branes[b] = self.branes[b][2:current_index] + [
                            self.branes[b][1]
                        ]
                        # if cancellation happened, there is a chance
                        # the provious point can be cancelled now
                        # also the numeration shifts by 1, because
                        # we removed two elements in the beginning
                        # [one completely removed, another moved to the end]
                        current_index -= 3
                    else:
                        current_index += 1

    def apply_braid(self, braid: list):
        for b in braid:
            self.apply_s(b)

    #    def tensor_product_index(self, indices : list):
    #        """
    #        Indices in a lexmin order on the tensor product basis
    #        """
    #        assert len(indices) == len(self.branes)
    #        for ind,br in zip(indices,self.branes):
    #            assert ind < len(br)
    #
    #        tensor_index = 0
    #        for ind, br in zip(indices[-1::-1],self.branes[-1::-1]):
    #            tensor_index *= len(br)
    #            tensor_index += ind
    #
    #        return tensor_index

    def one_dimentional_differential_initial(self, i):
        brane = self.branes[i]

        d0_dict = {}
        for current_index in range(len(brane)):
            if current_index != len(brane) - 1:
                next_index = current_index + 1
                have_endpoint = False
            # if we are at the end of line, we return to the beginning
            else:
                next_index = 0
                have_endpoint = True

            current_point = brane[current_index]
            next_point = brane[next_index]

            next_hom_deg = self.intersections_data[next_point].hom_deg
            next_segment = self.intersections_data[next_point].segment
            next_state = self.klrw_algebra[1].state(
                self.V if i == next_segment else self.W for i in range(self.n + 1)
            )
            current_hom_deg = self.intersections_data[current_point].hom_deg
            current_segment = self.intersections_data[current_point].segment
            current_state = self.klrw_algebra[1].state(
                self.V if i == current_segment else self.W for i in range(self.n + 1)
            )
            # shift by 1 in segments because numbering
            # of strands in KLRWAlgebra starts with 1.

            if next_hom_deg == current_hom_deg - 1:
                KLRWbraid = (
                    self.klrw_algebra[1]
                    .braid_set()
                    .braid_for_one_strand(
                        right_state=current_state,
                        right_moving_strand_position=current_segment + 1,
                        left_moving_strand_position=next_segment + 1,
                        check=False,
                    )
                )

                next_equ_deg = self.intersections_data[next_point].equ_deg
                current_equ_deg = self.intersections_data[current_point].equ_deg
                braid_degree = self.klrw_algebra[1].braid_degree(KLRWbraid)
                assert next_equ_deg - current_equ_deg == braid_degree, (
                    current_equ_deg.__repr__()
                    + " "
                    + next_equ_deg.__repr__()
                    + " "
                    + braid_degree.__repr__()
                )

                KLRWElement = self.klrw_algebra[1].monomial(KLRWbraid)
                if have_endpoint:
                    # we have sign (-1)^{(n/2)+1}
                    if len(brane) % 4 == 0:
                        # ???modify the sign?
                        KLRWElement *= ZZ(-1)
                d0_dict[next_index, current_index] = KLRWElement
            elif next_hom_deg == current_hom_deg + 1:
                KLRWbraid = (
                    self.klrw_algebra[1]
                    .braid_set()
                    .braid_for_one_strand(
                        right_state=next_state,
                        right_moving_strand_position=next_segment + 1,
                        left_moving_strand_position=current_segment + 1,
                        check=False,
                    )
                )

                next_equ_deg = self.intersections_data[next_point].equ_deg
                current_equ_deg = self.intersections_data[current_point].equ_deg
                braid_degree = self.klrw_algebra[1].braid_degree(KLRWbraid)
                assert current_equ_deg - next_equ_deg == braid_degree, (
                    current_equ_deg.__repr__()
                    + " "
                    + next_equ_deg.__repr__()
                    + " "
                    + braid_degree.__repr__()
                )

                KLRWElement = self.klrw_algebra[1].monomial(KLRWbraid)
                if have_endpoint:
                    # we have sign (-1)^{(n/2)+1}
                    if len(brane) % 4 == 0:
                        # ???modify the sign?
                        KLRWElement *= ZZ(-1)
                d0_dict[current_index, next_index] = KLRWElement
            else:
                raise ValueError("Cohomological degrees differ by an unexpected value")

        d0_csc = CSC_Mat.from_dict(
            d0_dict, number_of_rows=len(brane), number_of_columns=len(brane)
        )

        return d0_csc

    def one_dimentional_differential_corrections(self, i, order):
        brane = self.branes[i]

        # we organize points/T-thimbles in the brane by their cohomological degree as
        # a dictionary {hom_deg:(index_in_brane,TThimble)}
        points_by_hom_degree = {}
        for index in range(len(brane)):
            point = brane[index]
            thimble = self.intersections_data[point]
            hom_deg = thimble.hom_deg

            if hom_deg in points_by_hom_degree:
                points_by_hom_degree[hom_deg].append((index, thimble))
            else:
                points_by_hom_degree[hom_deg] = [(index, thimble)]

        d1_dict = {}
        variable_index = 0
        for hom_deg in sorted(points_by_hom_degree.keys()):
            # terms of differential exist only between adjacent coh degrees
            if hom_deg + 1 in points_by_hom_degree:
                thimbles_of_hom_deg = points_by_hom_degree[hom_deg]
                thimbles_of_hom_deg_plus_one = points_by_hom_degree[hom_deg + 1]

                for index0, thimble0 in thimbles_of_hom_deg:
                    for index1, thimble1 in thimbles_of_hom_deg_plus_one:
                        right_state = self.klrw_algebra[1].state(
                            self.V if i == thimble1.segment else self.W
                            for i in range(self.n + 1)
                        )
                        left_state = self.klrw_algebra[1].state(
                            self.V if i == thimble0.segment else self.W
                            for i in range(self.n + 1)
                        )
                        equivariant_degree = thimble0.equ_deg - thimble1.equ_deg

                        braid_degree = abs(thimble1.segment - thimble0.segment)
                        # we want to have exactly :order: dots
                        # each dot has degree 2
                        if 2 * order == equivariant_degree - braid_degree:
                            graded_component = self.klrw_algebra[1][
                                left_state:right_state:equivariant_degree
                            ]
                            basis = graded_component.basis()

                            if basis:
                                assert (
                                    len(basis) <= 1
                                ), "Too many possible corrections for one strand case."
                                # this is the only element
                                elem = basis[0]
                                # for braid, _ in elem:
                                #     assert (
                                #         self.klrw_algebra[1].braid_degree(braid)
                                #         == braid_degree
                                #     ), (
                                #         repr(self.klrw_algebra[1].braid_degree(braid))
                                #         + " "
                                #         + repr(braid_degree)
                                #     )

                                d1_dict[index0, index1] = {variable_index: elem}
                                variable_index += 1

        d1_csc = CSC_Mat.from_dict(
            d1_dict, number_of_rows=len(brane), number_of_columns=len(brane)
        )

        return d1_csc

    def differential_u_corrections(self, thimbles, max_number_of_dots, k, d_csc):
        # we organize points/T-thimbles in the brane by their cohomological degree as
        # a dictionary {hom_deg:(index_in_brane,TThimble)}
        points_by_hom_degree = {}
        for index, thimble in thimbles.items():
            hom_deg = thimble.hom_deg

            if hom_deg in points_by_hom_degree:
                points_by_hom_degree[hom_deg].append((index, thimble))
            else:
                points_by_hom_degree[hom_deg] = [(index, thimble)]

        print("Homological degrees:", len(points_by_hom_degree))
        for d, l in points_by_hom_degree.items():
            print(d, " : ", len(l))

        d1_dict = {}
        variable_index = 0
        for hom_deg in sorted(points_by_hom_degree.keys()):
            # terms of differential exist only between adjacent coh degrees
            if hom_deg + 1 in points_by_hom_degree:
                thimbles_of_hom_deg = points_by_hom_degree[hom_deg]
                thimbles_of_hom_deg_plus_one = points_by_hom_degree[hom_deg + 1]

                variables_of_degree = 0

                for index0, thimble0 in thimbles_of_hom_deg:
                    for index1, thimble1 in thimbles_of_hom_deg_plus_one:
                        # right_state = frozenset(thimble1.colored_state)
                        # left_state = frozenset(thimble0.colored_state)
                        right_state = self.klrw_algebra[k].state(
                            self.V if i in thimble1.colored_state else self.W
                            for i in range(self.n + k)
                        )
                        left_state = self.klrw_algebra[k].state(
                            self.V if i in thimble0.colored_state else self.W
                            for i in range(self.n + k)
                        )
                        equivariant_degree = thimble0.equ_deg - thimble1.equ_deg

                        graded_component = self.klrw_algebra[k][
                            left_state:right_state:equivariant_degree
                        ]
                        basis = list(
                            graded_component.basis(
                                max_number_of_dots=max_number_of_dots
                            ).values()
                        )

                        existing_entry = d_csc[index0, index1]

                        if not basis:
                            continue

                        """
                        if basis and existing_entry:
                            for braid, poly in existing_entry:
                                dots_algebra = self.klrw_algebra[k].base()
                                hom = dots_algebra.hom_ignore_non_dots()
                                only_dots_poly = hom(poly)
                                for dots_exp, _ in only_dots_poly.iterator_exp_coeff():
                                    existing_term = self.klrw_algebra[k].term(
                                        index=braid,
                                        coeff=dots_algebra.monomial(*dots_exp),
                                    )
                                    try:
                                        basis.remove(existing_term)
                                    except ValueError:
                                        pass  # we ignore if the term is not found
                        """
                        not_geometric_entry = True
                        if existing_entry is not None:
                            for _, coeff in existing_entry:
                                if not coeff.constant_coefficient().is_zero():
                                    not_geometric_entry = False
                                    break

                        if not_geometric_entry:
                            # only if we have new elements
                            d1_dict[index0, index1] = {}
                            for elem in basis:
                                d1_dict[index0, index1][variable_index] = elem
                                variable_index += 1
                                variables_of_degree += 1
                print(hom_deg, "::", variables_of_degree)

        d1_csc = CSC_Mat.from_dict(
            d1_dict, number_of_rows=len(thimbles), number_of_columns=len(thimbles)
        )

        return d1_csc, variable_index

    def one_dimentional_differential(self, i, method="pypardiso"):
        print("----------Making the initial approximation----------")
        d0_csc = self.one_dimentional_differential_initial(i)

        S = Solver(self.klrw_algebra[1])
        S.set_d0(d0_csc)

        for order in count(start=1):
            if S.d0().squares_to_zero():
                break
            print(
                "----------Correcting order {} ".format(order)
                + "in u for brane {}----------".format(i)
            )
            d1_csc = self.one_dimentional_differential_corrections(i, order=order)
            S.set_d1(d1_csc, number_of_variables=len(d1_csc._data()))
            # u = self.klrw_algebra[1].base().variables[self.V, self.W].monomial
            # multiplier = u**order
            multiplier = self.klrw_algebra[1].base().one()
            S.make_corrections(
                multiplier=multiplier,
                order=order,
                graded_type="u^order*h^0",
                method=method,
            )

        return S.d0()

    def make_differential(
        self,
        max_number_of_dots=2,
        max_order_in_hbar=5,
        method="pypardiso",
    ):
        assert self.branes, "There has to be at least one E-brane."

        # assign all intersection points order, i.e. the number
        # of other intersection points to the left
        ind = 0
        for seg in self.intersections:
            for pt in seg:
                self.intersections_data[pt].order = ind
                ind = ind + 1

        # the differential for the zeroth brane
        d_csc_current = self.one_dimentional_differential(0, method=method)

        # print(np.asarray(d_csc_current._data()))
        # print(np.asarray(d_csc_current._indices()))
        # print(np.asarray(d_csc_current._indptrs()))
        # print(d_csc_current._number_of_rows())

        thimbles = {}
        for index, pt in zip(count(), self.branes[0]):
            # we make the key a tuple for consistency with later iterarions
            thimbles[index] = ProductThimbles(
                colored_state=(self.intersections_data[pt].segment,),
                hom_deg=self.intersections_data[pt].hom_deg,
                equ_deg=self.intersections_data[pt].equ_deg,
                order=(self.intersections_data[pt].order,),
                next_thimble_strand_number=1,
                next_thimble_position=self.intersections_data[pt].segment + 1,
                intersection_indices=(pt,),
            )

        # for degree test
        indptr: cython.int
        indptr_end: cython.int
        for j in range(len(thimbles)):
            column_thimble = thimbles[j]

            indptr: cython.int = d_csc_current._indptrs()[j]
            indptr_end: cython.int = d_csc_current._indptrs()[j + 1]
            while indptr != indptr_end:
                i: cython.int = d_csc_current._indices()[indptr]
                row_thimble = thimbles[i]
                entry = d_csc_current._data()[indptr]
                assert (
                    column_thimble.hom_deg == row_thimble.hom_deg + 1
                ), "Cohomological degrees differ by an unexpected value"
                assert row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(
                    check_if_homogeneous=True
                ), (
                    row_thimble.equ_deg.__repr__()
                    + " "
                    + column_thimble.equ_deg.__repr__()
                    + " "
                    + entry.degree().__repr__()
                )
                indptr += 1

        # on each step we add one more brane and correct the result
        # we call _current any things from the differential of all the branes
        # before the step and _next for the one-strand differential of the new brane
        for next_brane_number in range(1, len(self.branes)):
            thimbles_current = thimbles.copy()
            thimbles_next = {}
            for index, pt in zip(count(), self.branes[next_brane_number]):
                thimbles_next[index] = ProductThimbles(
                    colored_state=(self.intersections_data[pt].segment,),
                    hom_deg=self.intersections_data[pt].hom_deg,
                    equ_deg=self.intersections_data[pt].equ_deg,
                    order=(self.intersections_data[pt].order,),
                    next_thimble_strand_number=1,
                    next_thimble_position=self.intersections_data[pt].segment + 1,
                    intersection_indices=(pt,),
                )

            # the next_brane_diff is the differential for the next_brane_number-th brane
            d_csc_next = self.one_dimentional_differential(
                next_brane_number, method=method
            )

            # for degree test
            indptr: cython.int
            indptr_end: cython.int
            for j in range(len(thimbles_next)):
                column_thimble = thimbles_next[j]

                indptr: cython.int = d_csc_next._indptrs()[j]
                indptr_end: cython.int = d_csc_next._indptrs()[j + 1]
                while indptr != indptr_end:
                    i: cython.int = d_csc_next._indices()[indptr]
                    row_thimble = thimbles_next[i]
                    entry = d_csc_next._data()[indptr]
                    assert (
                        column_thimble.hom_deg == row_thimble.hom_deg + 1
                    ), "Cohomological degrees differ by an unexpected value"
                    assert row_thimble.equ_deg - column_thimble.equ_deg == entry.degree(
                        check_if_homogeneous=True
                    ), (
                        row_thimble.equ_deg.__repr__()
                        + " "
                        + column_thimble.equ_deg.__repr__()
                        + " "
                        + entry.degree().__repr__()
                    )
                    indptr += 1

            thimbles = {}
            index = 0
            for index_current in thimbles_current:
                thimble_current = thimbles_current[index_current]
                point_orders_current_sorted = sorted(thimble_current.order)
                # print(point_orders_current_sorted)
                for index_next in thimbles_next:
                    thimble_next = thimbles_next[index_next]
                    order = thimble_current.order + thimble_next.order
                    hom_deg = thimble_current.hom_deg + thimble_next.hom_deg
                    intersection_indices = (
                        thimble_current.intersection_indices
                        + thimble_next.intersection_indices
                    )

                    # thimble_next.order and thimble_next.colored_state
                    # have one element each
                    order_next = thimble_next.order[0]
                    state_next = thimble_next.colored_state[0]

                    assert order_next not in point_orders_current_sorted

                    # print(thimble_current, thimble_next)

                    # finding the position of the next brane among the current branes
                    strands_to_the_left = bisect.bisect_left(
                        point_orders_current_sorted, order_next
                    )

                    # Now we know that the position of the last thimble
                    # is segment_next+i
                    # All points to the right of it have to add one to
                    # their strand number
                    position_next = state_next + strands_to_the_left
                    colored_state_list = list(thimble_current.colored_state)
                    for i in range(len(colored_state_list)):
                        if colored_state_list[i] >= position_next:
                            colored_state_list[i] += 1
                    colored_state_list.append(position_next)
                    colored_state = tuple(colored_state_list)
                    # print(colored_state)
                    # the equivariant degree is the sum of the equivariant degrees
                    # of 1-d thimbles
                    # and the contributions for each pair
                    # the pair contributions inside of current are already taken into
                    # account in thimble_current.equ_deg
                    equ_deg = thimble_current.equ_deg + thimble_next.equ_deg
                    # the next thimble is 1-d, so only one point
                    pt_next = thimble_next.intersection_indices[0]
                    for pt_current in thimble_current.intersection_indices:
                        equ_deg += self.pairwise_equ_deg[pt_current, pt_next]

                    thimbles[index] = ProductThimbles(
                        colored_state=colored_state,
                        hom_deg=hom_deg,
                        equ_deg=equ_deg,
                        order=order,
                        next_thimble_strand_number=strands_to_the_left + 1,
                        next_thimble_position=position_next,
                        intersection_indices=intersection_indices,
                    )
                    # print(thimble_current,thimble_next)
                    # print(index, " : ", thimbles[index])
                    index += 1

            # print(thimbles)

            # KLRW algebra on next_brane_number+1 strands
            klrw_algebra = self.klrw_algebra[next_brane_number + 1]

            # print(np.asarray(d_csc_next._data()))
            # print(np.asarray(d_csc_next._indices()))
            # print(np.asarray(d_csc_next._indptrs()))
            # print(d_csc_next._number_of_rows())

            # we are looking for the differential that is the tensor product of diff
            # and next_brane_diff + corrections

            number_of_columns_current: cython.int = d_csc_current._number_of_rows()
            number_of_entries_current: cython.int = d_csc_current.nnz()
            number_of_columns_next: cython.int = d_csc_next._number_of_rows()
            number_of_entries_next: cython.int = d_csc_next.nnz()
            # each non-zero element in d_csc gives number_of_columns_next terms
            # each non-zero element in d_csc_next_brane gives number_of_columns_current
            # terms in the differential of form d \otimes 1 + (-1)^{...} 1 \otimes d
            # because of the homological gradings no terms are
            # in the same matrix element
            number_of_entries: cython.int = (
                number_of_entries_current * number_of_columns_next
                + number_of_entries_next * number_of_columns_current
            )
            number_of_columns: cython.int = (
                number_of_columns_current * number_of_columns_next
            )

            d0_csc_data = np.empty(number_of_entries, dtype="O")
            d0_csc_indices = np.zeros(number_of_entries, dtype="intc")
            d0_csc_indptrs = np.zeros(number_of_columns + 1, dtype="intc")

            j_current: cython.int
            j_next: cython.int
            # i_current : cython.int
            # i_next : cython.int
            indptr_current: cython.int
            indptr_next: cython.int
            indptr_end_current: cython.int
            indptr_end_next: cython.int
            entries_so_far: cython.int = 0
            for j_current in range(number_of_columns_current):
                # for signs later we remember cohomological degree
                cur_hom_degree = thimbles_current[j_current].hom_deg

                for j_next in range(number_of_columns_next):
                    # we write the column with the index (j_current,j_next), i.e.
                    # j_current*number_of_columns_next + j_next
                    # TODO: can make one case if we sort thimbles by cohomological
                    # degree first?

                    column_index = j_current * number_of_columns_next + j_next
                    column_thimble = thimbles[column_index]
                    column_next_position = column_thimble.next_thimble_position
                    column_colored_state = column_thimble.colored_state
                    column_state = klrw_algebra.state(
                        self.V if i in column_colored_state else self.W
                        for i in range(self.n + next_brane_number + 1)
                    )
                    # column_state = frozenset(
                    #     column_colored_state
                    # )

                    indptr_current = d_csc_current._indptrs()[j_current]
                    indptr_next = d_csc_next._indptrs()[j_next]
                    indptr_end_current = d_csc_current._indptrs()[j_current + 1]
                    indptr_end_next = d_csc_next._indptrs()[j_next + 1]

                    def add_d_times_one_term(indptr_current, entries_so_far):
                        row_index = (
                            d_csc_current._indices()[indptr_current]
                            * number_of_columns_next
                            + j_next
                        )
                        row_thimble = thimbles[row_index]
                        row_next_position = row_thimble.next_thimble_position
                        # row_colored_state = row_thimble.colored_state
                        # row_state = klrw_algebra.state(
                        #     self.V if i in row_colored_state else self.W
                        #     for i in range(self.n + next_brane_number + 1)
                        # )
                        # row_state = frozenset(
                        #     row_colored_state
                        # )
                        # left_subset_of_strands = frozenset(
                        #     range(1, next_brane_number + 2)
                        # ) - frozenset((row_thimble.next_thimble_strand_number,))
                        left_new_strand_number = row_thimble.next_thimble_strand_number
                        hom_to_more_dots = self.hom_add_one_more_strand[
                            next_brane_number, left_new_strand_number
                        ]

                        entry_current = d_csc_current._data()[indptr_current]

                        entry = klrw_algebra.zero()
                        for braid, coef in entry_current:
                            # prepare the new braid by adding one more strand
                            mapping = {}
                            for t in column_colored_state:
                                if t != column_next_position:
                                    if t > column_next_position:
                                        t_cur = t - 1
                                    else:
                                        t_cur = t
                                    b = braid.find_position_on_other_side(
                                        t_cur, reverse=True
                                    )
                                    if b >= row_next_position:
                                        b += 1
                                    mapping[b] = t
                            d_times_one_braid = (
                                klrw_algebra.braid_set().braid_by_extending_permutation(
                                    right_state=column_state,
                                    mapping=mapping,
                                )
                            )
                            new_coeff = hom_to_more_dots(coef)
                            if not new_coeff.is_zero():
                                braid_degree = klrw_algebra.braid_degree(
                                    d_times_one_braid
                                )
                                coeff_degree = klrw_algebra.base().element_degree(
                                    new_coeff, check_if_homogeneous=True
                                )
                                term_degree = braid_degree + coeff_degree
                            if (
                                row_thimble.equ_deg - column_thimble.equ_deg
                                == term_degree
                            ):
                                entry += klrw_algebra.term(d_times_one_braid, new_coeff)

                        assert (
                            column_thimble.hom_deg == row_thimble.hom_deg + 1
                        ), "Cohomological degrees differ by an unexpected value"

                        if not entry.is_zero():
                            d0_csc_indices[entries_so_far] = row_index
                            d0_csc_data[entries_so_far] = entry
                            return entries_so_far + 1
                        else:
                            return entries_so_far

                    def add_one_times_d_term(indptr_next, entries_so_far):
                        row_index = (
                            j_current * number_of_columns_next
                            + d_csc_next._indices()[indptr_next]
                        )
                        row_thimble = thimbles[row_index]
                        row_next_position = row_thimble.next_thimble_position
                        # row_colored_state = row_thimble.colored_state
                        # row_state = klrw_algebra.state(
                        #     self.V if i in row_colored_state else self.W
                        #     for i in range(self.n + next_brane_number + 1)
                        # )
                        # row_state = frozenset(
                        #    row_colored_state
                        # )
                        # left_subset_of_strands = frozenset(
                        #    (row_thimble.next_thimble_strand_number,)
                        # )
                        left_new_strand_number = row_thimble.next_thimble_strand_number
                        hom_to_more_dots = self.hom_one_to_many_dots[
                            next_brane_number, left_new_strand_number
                        ]

                        entry_next = d_csc_next._data()[indptr_next]

                        entry = klrw_algebra.zero()
                        # there is only one possible braid with one moving strand
                        for braid, coef in entry_next:
                            # prepare the new braid by adding one more strand
                            mapping = {row_next_position: column_next_position}
                            # ??? More efficient
                            one_times_d_braid = (
                                klrw_algebra.braid_set().braid_by_extending_permutation(
                                    right_state=column_state,
                                    mapping=mapping,
                                )
                            )
                            new_coeff = hom_to_more_dots(coef)
                            entry += klrw_algebra.term(one_times_d_braid, new_coeff)

                        assert (
                            column_thimble.hom_deg == row_thimble.hom_deg + 1
                        ), "Cohomological degrees differ by an unexpected value"
                        if (
                            row_thimble.equ_deg - column_thimble.equ_deg
                            == entry.degree(check_if_homogeneous=True)
                        ):
                            d0_csc_indices[entries_so_far] = row_index
                            if cur_hom_degree % 2 == 0:
                                d0_csc_data[entries_so_far] = entry
                            else:
                                d0_csc_data[entries_so_far] = -entry
                            return entries_so_far + 1
                        else:
                            return entries_so_far

                    # for this range of row indices in d_csc_current
                    # only d \otimes 1 contributes
                    while indptr_current != indptr_end_current:
                        if d_csc_current._indices()[indptr_current] < j_current:
                            entries_so_far = add_d_times_one_term(
                                indptr_current, entries_so_far
                            )
                            indptr_current += 1
                        else:
                            break

                    # terms of form (-1)^... 1 \times d
                    while indptr_next != indptr_end_next:
                        if d_csc_next._indices()[indptr_next] < j_next:
                            entries_so_far = add_one_times_d_term(
                                indptr_next, entries_so_far
                            )
                            indptr_next += 1
                        else:
                            break

                    # one term from d \otimes 1
                    if indptr_current != indptr_end_current:
                        if d_csc_current._indices()[indptr_current] == j_current:
                            entries_so_far = add_d_times_one_term(
                                indptr_current, entries_so_far
                            )
                            indptr_current += 1

                    # remaining terms of form (-1)^... 1 \times d
                    while indptr_next != indptr_end_next:
                        entries_so_far = add_one_times_d_term(
                            indptr_next, entries_so_far
                        )
                        indptr_next += 1

                    # now row indices in d_csc_current are >j_current,
                    # again only d \otimes 1 contributes
                    while indptr_current != indptr_end_current:
                        entries_so_far = add_d_times_one_term(
                            indptr_current, entries_so_far
                        )
                        # entries_so_far += 1
                        indptr_current += 1

                    d0_csc_indptrs[column_index + 1] = entries_so_far

            d0_csc_indices = d0_csc_indices[:entries_so_far]
            d0_csc_data = d0_csc_data[:entries_so_far]

            for i in range(len(d0_csc_data)):
                if d0_csc_data[i].is_zero():
                    print("Zero element in the possible corrections matrix:", i)
            d_csc = CSC_Mat(
                data=d0_csc_data,
                indices=d0_csc_indices,
                indptrs=d0_csc_indptrs,
                number_of_rows=number_of_columns,
            )

            S = Solver(klrw_algebra)
            S.set_d0(d_csc)

            d1_csc, number_of_variables = self.differential_u_corrections(
                thimbles=thimbles,
                max_number_of_dots=max_number_of_dots,
                k=next_brane_number + 1,
                d_csc=S.d0(),
            )

            from pickle import dump

            with open("d1_local", "wb") as f:
                dump(d1_csc.dict(), file=f)

            S.set_d1(d1_csc, number_of_variables=number_of_variables)
            # u = klrw_algebra.base().variables[self.V, self.W].monomial
            h = klrw_algebra.base().variables[self.V].monomial
            for order in range(1, max_order_in_hbar):
                print(
                    "----------Correcting order {} in h ".format(order)
                    + "for the product of the first {} branes----------".format(
                        next_brane_number + 1
                    )
                )
                # multiplier = (h * u) ** order
                multiplier = h**order
                S.make_corrections(
                    multiplier=multiplier,
                    order=order,
                    graded_type="h^order",
                    method=method,
                )
                if S.d0().squares_to_zero():
                    break
            else:
                raise RuntimeError(
                    "The system does not square to zero."
                    + "Increase the order in hbar or max number of dots."
                )

            d_csc_current = S.d0()

        self.differential = d_csc_current
        self.thimbles = thimbles

    # @cython.ccall
    def find_differential_matrix(
        self,
        domain_indices: list,
        codomain_indices: list,
        R: PrincipalIdealDomain,
        dualize_complex=False,
    ) -> Matrix_sparse:
        from .framed_dynkin import is_a_dot_index

        dots_algebra = self.klrw_algebra[self.k].base()

        variables_images = []
        for index in dots_algebra.variables:
            if dots_algebra.variables[index].name is not None:
                if is_a_dot_index(index):
                    # setting all dots to zero
                    variables_images.append(R.zero())
                else:
                    # setting all other parameters to 1
                    variables_images.append(R.one())

        hom = dots_algebra.hom(variables_images, R)

        d_dict = {}
        for ind_j, j in enumerate(domain_indices):
            for ind_i, i in enumerate(codomain_indices):
                if dualize_complex:
                    elem = self.differential[j, i]
                else:
                    elem = self.differential[i, j]
                if elem is not None:
                    # no_terms_found = True
                    for braid, coef in elem:
                        if braid.word() == ():
                            d_dict[ind_i, ind_j] = hom(coef)

        d_mat = matrix(
            R,
            ncols=len(domain_indices),
            nrows=len(codomain_indices),
            entries=d_dict,
            sparse=True,
        )
        return d_mat

    def part_of_graded_component(
        self,
        relevant_thimbles,
        current_hom_deg,
        current_equ_deg,
        i,
    ):
        if i == len(relevant_thimbles):
            return False
        if relevant_thimbles[i][1].hom_deg != current_hom_deg:
            return False
        if relevant_thimbles[i][1].equ_deg != current_equ_deg:
            return False
        return True

    def find_homology(
        self,
        R: PrincipalIdealDomain,
        hom_deg_shift=0,
        equ_deg_shift=0,
        dualize_complex=False,
    ):
        """
        Working over R that is a PID.
        [works on fields and integers, modules over
        other PIDs not fully implemented in Sage yet]
        Returns a Poincare Polynomial if R is a field
        and a dictionary {degrees:invariant factors}
        if R is the ring of integers
        """

        # {2,4,6,...} in Elise's convention
        # {1,4,7,...} in our convention
        relevant_uncolored_state = frozenset(1 + 3 * i for i in range(self.k))

        relevant_thimbles = [
            (i, st)
            for i, st in self.thimbles.items()
            if st.uncolored_state() == relevant_uncolored_state
        ]

        print("The number of relevant thimbles:", len(relevant_thimbles))

        # here we will use that it our convention differential
        # *preserves* the equivariant degree
        # and decreases the homological degree by 1.
        if not dualize_complex:
            relevant_thimbles.sort(key=(lambda x: (x[1].equ_deg, -x[1].hom_deg)))
            diff_hom_deg = -1
        else:
            relevant_thimbles.sort(key=(lambda x: (x[1].equ_deg, x[1].hom_deg)))
            diff_hom_deg = 1

        d_prev: Matrix_sparse
        d_next: Matrix_sparse
        # C1_indices: list
        C2_indices: list
        C3_indices: list

        # we will find all triples C1, C2, C3 that they enter as
        # C1->C2->C3 in the complex, possibly with zero C1 or C3.
        # Then we find the maps -> which we call d_prev and d_next
        # and compute the homology at C2.
        if R.is_field():
            LauPoly = LaurentPolynomialRing(ZZ, 2, ["t", "q"])
            t = LauPoly("t")
            q = LauPoly("q")
            PoincarePolynomial = LauPoly(0)
        else:
            Homology = {}
        i = 0
        # current_hom_deg = 0
        C3_indices = []
        while i != len(relevant_thimbles):
            # if the chain did not break by ending as ->0 on the previous step
            if not C3_indices:
                # C1_indices = []
                C2_indices = []
                # degrees of C2
                current_hom_deg = relevant_thimbles[i][1].hom_deg
                current_equ_deg = relevant_thimbles[i][1].equ_deg

                while self.part_of_graded_component(
                    relevant_thimbles,
                    current_hom_deg,
                    current_equ_deg,
                    i,
                ):
                    C2_indices.append(relevant_thimbles[i][0])
                    i += 1

                # we already know C3_indices = [] from if
                while self.part_of_graded_component(
                    relevant_thimbles,
                    current_hom_deg + diff_hom_deg,
                    current_equ_deg,
                    i,
                ):
                    C3_indices.append(relevant_thimbles[i][0])
                    i += 1

                d_next = self.find_differential_matrix(
                    C2_indices, C3_indices, R, dualize_complex=dualize_complex
                )
                # print(C1_indices, "->", C2_indices, "->", C3_indices)
                # print(d_next)
                # print("-----")

                if R.is_field():
                    PoincarePolynomial += (
                        d_next.right_nullity() * t**current_hom_deg * q**current_equ_deg
                    )

                # over a PID a submodule of a free module is free
                else:
                    homology_group = d_next.right_kernel()
                    invariants = [0] * homology_group.rank()
                    if invariants:
                        Homology[
                            current_hom_deg + current_equ_deg + hom_deg_shift,
                            current_equ_deg + equ_deg_shift,
                        ] = [R.quotient(inv * R) for inv in invariants]
            else:
                # C1_indices = C2_indices
                d_prev = d_next.__copy__()
                current_hom_deg += diff_hom_deg
                C2_indices = C3_indices

                C3_indices = []
                while self.part_of_graded_component(
                    relevant_thimbles,
                    current_hom_deg + diff_hom_deg,
                    current_equ_deg,
                    i,
                ):
                    C3_indices.append(relevant_thimbles[i][0])
                    i += 1

                d_next = self.find_differential_matrix(
                    C2_indices, C3_indices, R, dualize_complex=dualize_complex
                )

                assert (d_next * d_prev).is_zero()
                # print(C1_indices, "->", C2_indices, "->", C3_indices)
                # print(d_prev, "\n")
                # print(d_next)
                # print("-----")

                if R.is_field():
                    PoincarePolynomial += (
                        (d_next.right_nullity() - d_prev.rank())
                        * t**current_hom_deg
                        * q**current_equ_deg
                    )
                else:
                    homology_group = d_next.right_kernel() / d_prev.column_module()
                    invariants = homology_group.invariants()
                    if invariants:
                        Homology[
                            current_hom_deg + current_equ_deg + hom_deg_shift,
                            current_equ_deg + equ_deg_shift,
                        ] = [R.quotient(inv * R) for inv in invariants]

        if R.is_field():
            return PoincarePolynomial
        else:
            return Homology
