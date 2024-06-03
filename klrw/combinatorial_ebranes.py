from collections import defaultdict
from typing import Iterable, NamedTuple

from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.rings.ring import PrincipalIdealDomain
from sage.matrix.constructor import matrix
from sage.combinat.root_system.cartan_type import CartanType

from .klrw_algebra import KLRWAlgebra
from .framed_dynkin import FramedDynkinDiagram_with_dimensions
from .klrw_state import KLRWstate
from .perfect_complex import KLRWProjectiveModule, KLRWPerfectComplex
from .perfect_complex_corrections import corrected_diffirential_csc
from klrw.cython_exts.combinatorial_ebranes_ext import (
    d_times_one_piece,
    one_times_d_piece,
    CSC_from_dict_of_blocks,
)


class TThimble(NamedTuple):
    segment: int
    hom_deg: int
    equ_deg: int


class ProductThimbles(NamedTuple):
    """
    colored_state is an Iterable where i-th element is the segment of the i-th
    moving strand
    points is an Iterable where i-th element is the number of an intersecion
    point corresponding to i-th strand
    """

    colored_state: Iterable
    points: Iterable


class CombinatorialEBrane:
    def __init__(self, number_of_punctures, number_of_E_branes):
        self.n = number_of_punctures
        self.k = number_of_E_branes

        quiver_data = FramedDynkinDiagram_with_dimensions(CartanType(["A", 1]))
        self.quiver = quiver_data.quiver
        self.V, self.W = self.quiver.vertices()
        quiver_data[self.W] = self.n

        # we will use several KLRW algebras
        # one for each of the subset of the moving strands
        self.klrw_algebra = {}
        for i in range(1, self.k + 1):
            quiver_data[self.V] = i
            self.klrw_algebra[i] = KLRWAlgebra(
                ZZ,
                quiver_data,
                vertex_scaling=True,
                edge_scaling=True,
                vertex_prefix="h",
                framing_prefix="u",
                warnings=True,
            )
            # self.klrw_algebra[i].braid_set().enable_checks()
            # print("****")

        # We need to convert elements of one algebra of dots
        # to elements of the one with more strands.
        # self.hom_one_to_many_dots[i,j] is the homomorphism
        # from the algebra of dots on one strand to the algebra
        # on i + 1 strands that sends all dots to dots on j-th strand.
        self.hom_one_to_many_dots = {}
        domain_dots_algebra = self.klrw_algebra[1].base()
        for i in range(1, self.k):
            self.hom_one_to_many_dots[i] = {}
            codomain_dots_algebra = self.klrw_algebra[i + 1].base()
            for j in range(1, i + 2):
                # dots on the only strand go to dots on j-th strand
                map = {(self.V, 1): (self.V, j)}
                hom = domain_dots_algebra.hom_from_dots_map(codomain_dots_algebra, map)
                self.hom_one_to_many_dots[i][j] = hom

        # self.hom_add_one_more_strand[i,j] is the homomorphism
        # from the algebra of dots on i strands to the algebra
        # on i+1 strands keeping j-th strand without dots.
        self.hom_add_one_more_strand = {}
        for i in range(1, self.k):
            self.hom_add_one_more_strand[i] = {}
            domain_dots_algebra = self.klrw_algebra[i].base()
            codomain_dots_algebra = self.klrw_algebra[i + 1].base()
            for j in range(1, i + 2):
                # dots on the only strand go to dots on j-th strand
                map = {
                    (self.V, k): (self.V, k) if k < j else (self.V, k + 1)
                    for k in range(1, i + 1)
                }
                hom = domain_dots_algebra.hom_from_dots_map(codomain_dots_algebra, map)
                self.hom_add_one_more_strand[i][j] = hom

        E_brane_cyclic = [0, 1, 2, 3]
        E_brane_intersections = [[0], [1, 3], [2]]
        # Warning about equivariant grading: we picked the convention
        # from https://arxiv.org/pdf/0804.2080.pdf
        # The convention from https://arxiv.org/pdf/2305.13480.pdf
        # is asymmetric, i.e. depends on
        # if we read the braid right-to-lefttom or lefttom-to-right
        E_branes_intersections_data = {
            0: TThimble(segment=0, hom_deg=0, equ_deg=0),
            1: TThimble(segment=1, hom_deg=-1, equ_deg=1),
            2: TThimble(segment=2, hom_deg=0, equ_deg=0),
            3: TThimble(segment=1, hom_deg=1, equ_deg=-1),
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
                    )

        # keeps track of branch in log(yi-yj)
        # initially everything in the principal branch
        self.pairwise_equ_deg_halved = {}
        for b1 in range(len(self.branes)):
            for b0 in range(b1):
                for pt0 in self.branes[b0]:
                    for pt1 in self.branes[b1]:
                        self.pairwise_equ_deg_halved[pt0, pt1] = 0

        self.number_of_intersections = (
            number_of_E_branes * intersection_points_in_E_brane
        )

        # we use homological degree convention: differential
        # lowers homological grading by 1
        self.degree = -1

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
                    segment=i - 1, hom_deg=hom_deg, equ_deg=equ_deg - 1
                )
                self.intersections_data[right_point] = TThimble(
                    segment=i + 1, hom_deg=hom_deg, equ_deg=equ_deg - 1
                )
                self.intersections_data[point] = TThimble(
                    segment=i, hom_deg=hom_deg + 1, equ_deg=equ_deg - 2
                )
            else:
                self.intersections_data[left_point] = TThimble(
                    segment=i - 1, hom_deg=hom_deg, equ_deg=equ_deg + 1
                )
                self.intersections_data[right_point] = TThimble(
                    segment=i + 1, hom_deg=hom_deg, equ_deg=equ_deg + 1
                )
                self.intersections_data[point] = TThimble(
                    segment=i, hom_deg=hom_deg - 1, equ_deg=equ_deg + 2
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
        old_keys = [x for x in self.pairwise_equ_deg_halved]
        for pts in old_keys:
            # pts is a 2-tuple
            if_pt_on_the_segment = [p in self.intersections[i] for p in pts]
            # if lefth points are on the segment
            if if_pt_on_the_segment[0] and if_pt_on_the_segment[1]:
                # find the places of pt0,pt1 and corresponding left&right points
                js = [self.intersections[i].index(p) for p in pts]

                left_points = [new_points_left[j] for j in js]
                right_points = [new_points_right[j] for j in js]

                self.pairwise_equ_deg_halved[left_points[0], left_points[1]] = (
                    self.pairwise_equ_deg_halved[pts]
                )
                self.pairwise_equ_deg_halved[right_points[0], right_points[1]] = (
                    self.pairwise_equ_deg_halved[pts]
                )
                # for many cases the value depents on relative positions
                # of points on the segment
                if js[0] < js[1]:
                    self.pairwise_equ_deg_halved[left_points[0], pts[1]] = (
                        self.pairwise_equ_deg_halved[pts]
                    )
                    self.pairwise_equ_deg_halved[left_points[0], right_points[1]] = (
                        self.pairwise_equ_deg_halved[pts]
                    )
                    self.pairwise_equ_deg_halved[pts[0], left_points[1]] = (
                        self.pairwise_equ_deg_halved[pts] - sign
                    )
                    self.pairwise_equ_deg_halved[pts[0], right_points[1]] = (
                        self.pairwise_equ_deg_halved[pts]
                    )
                    self.pairwise_equ_deg_halved[right_points[0], left_points[1]] = (
                        self.pairwise_equ_deg_halved[pts] - sign
                    )
                    self.pairwise_equ_deg_halved[right_points[0], pts[1]] = (
                        self.pairwise_equ_deg_halved[pts] - sign
                    )
                else:
                    self.pairwise_equ_deg_halved[left_points[0], pts[1]] = (
                        self.pairwise_equ_deg_halved[pts] - sign
                    )
                    self.pairwise_equ_deg_halved[left_points[0], right_points[1]] = (
                        self.pairwise_equ_deg_halved[pts] - sign
                    )
                    self.pairwise_equ_deg_halved[pts[0], left_points[1]] = (
                        self.pairwise_equ_deg_halved[pts]
                    )
                    self.pairwise_equ_deg_halved[pts[0], right_points[1]] = (
                        self.pairwise_equ_deg_halved[pts] - sign
                    )
                    self.pairwise_equ_deg_halved[right_points[0], left_points[1]] = (
                        self.pairwise_equ_deg_halved[pts]
                    )
                    self.pairwise_equ_deg_halved[right_points[0], pts[1]] = (
                        self.pairwise_equ_deg_halved[pts]
                    )
                # we change this the last because it's used in all the other cases
                self.pairwise_equ_deg_halved[pts] += -sign

            elif if_pt_on_the_segment[0]:
                # and p1 is not on the segment from if statement above
                # find the place of pt0 and corresponding left&right points
                j = self.intersections[i].index(pts[0])
                left_point = new_points_left[j]
                right_point = new_points_right[j]
                # keep self.pairwise_equ_deg_halved[pt0,pt1] the same
                self.pairwise_equ_deg_halved[left_point, pts[1]] = (
                    self.pairwise_equ_deg_halved[pts]
                )
                self.pairwise_equ_deg_halved[right_point, pts[1]] = (
                    self.pairwise_equ_deg_halved[pts]
                )

            elif if_pt_on_the_segment[1]:
                # and p1 is not on the segment from if statement above
                # find the place of pt0 and corresponding left&right points
                j = self.intersections[i].index(pts[1])
                left_point = new_points_left[j]
                right_point = new_points_right[j]
                # keep self.pairwise_equ_deg_halved[pt0,pt1] the same
                self.pairwise_equ_deg_halved[pts[0], left_point] = (
                    self.pairwise_equ_deg_halved[pts]
                )
                self.pairwise_equ_deg_halved[pts[0], right_point] = (
                    self.pairwise_equ_deg_halved[pts]
                )

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

    def is_ordered(self):
        # checks if intersection points are enumerated from left to right
        current_index = 0
        for segment in self.intersections:
            for j in segment:
                if j != current_index:
                    return False
                current_index += 1
        return True

    def order_intersections(self):
        if self.is_ordered():
            return

        old_to_new_index = {}
        new_index = 0
        for segment in self.intersections:
            for j in range(len(segment)):
                old_index = segment[j]
                segment[j] = new_index
                old_to_new_index[old_index] = new_index
                new_index += 1

        new_intersections_data = {
            old_to_new_index[pt]: data for pt, data in self.intersections_data.items()
        }
        self.intersections_data = new_intersections_data

        for brane in self.branes:
            for j in range(len(brane)):
                brane[j] = old_to_new_index[brane[j]]

        new_pairwise_equ_deg_halved = {
            (
                old_to_new_index[pt1],
                old_to_new_index[pt2],
            ): deg
            for (pt1, pt2), deg in self.pairwise_equ_deg_halved.items()
        }
        self.pairwise_equ_deg_halved = new_pairwise_equ_deg_halved

    def one_dimensional_differential_geometric(self, i):
        brane = self.branes[i]
        projectives = defaultdict(list)
        thimbles = defaultdict(list)

        vw_crossing_degree = self.klrw_algebra[1].grading_group.crossing_grading(
            self.V, self.W
        )

        differential_dicts = defaultdict(dict)
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
            next_equ_deg = self.intersections_data[next_point].equ_deg
            next_segment = self.intersections_data[next_point].segment
            next_state = self.klrw_algebra[1].state(
                self.V if i == next_segment else self.W for i in range(self.n + 1)
            )
            current_hom_deg = self.intersections_data[current_point].hom_deg
            current_equ_deg = self.intersections_data[current_point].equ_deg
            current_segment = self.intersections_data[current_point].segment
            current_state = self.klrw_algebra[1].state(
                self.V if i == current_segment else self.W for i in range(self.n + 1)
            )

            thimbles[current_hom_deg].append(
                ProductThimbles(
                    colored_state=(current_segment,),
                    points=(current_point,),
                )
            )
            projectives[current_hom_deg].append(
                KLRWProjectiveModule(
                    state=current_state,
                    equivariant_degree=current_equ_deg * vw_crossing_degree,
                )
            )
            current_index_within_hom_deg = len(projectives[current_hom_deg]) - 1
            if not have_endpoint:
                # we will add this projective in the next run of this cycle
                next_index_within_hom_deg = len(projectives[next_hom_deg])
            else:
                next_index_within_hom_deg = 0

            if next_hom_deg == current_hom_deg + 1:
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

                braid_degree = self.klrw_algebra[1].braid_degree(KLRWbraid)
                assert (
                    current_equ_deg - next_equ_deg == braid_degree.ordinary_grading()
                ), (
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
                        KLRWElement *= ZZ(-1)

                differential_dicts[next_hom_deg][
                    next_index_within_hom_deg, current_index_within_hom_deg
                ] = KLRWElement
            elif next_hom_deg == current_hom_deg - 1:
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

                braid_degree = self.klrw_algebra[1].braid_degree(KLRWbraid)
                assert (
                    next_equ_deg - current_equ_deg == braid_degree.ordinary_grading()
                ), (
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
                        KLRWElement *= ZZ(-1)

                differential_dicts[current_hom_deg][
                    current_index_within_hom_deg, next_index_within_hom_deg
                ] = KLRWElement
            else:
                raise ValueError("Cohomological degrees differ by an unexpected value")

        return differential_dicts, projectives, thimbles

    def one_dimensional_differential_projectives_thimbles(
        self, i, parallel_processes=1
    ):
        differential_dicts, projectives, thimbles = (
            self.one_dimensional_differential_geometric(i)
        )

        diff_csc = corrected_diffirential_csc(
            self.klrw_algebra[1],
            differential_dicts,
            projectives,
            parallel_processes=parallel_processes,
            degree=self.degree,
        )

        return diff_csc, projectives, thimbles

    @staticmethod
    def hom_degrees_in_product(projectives_curr, projective_next):
        hom_degrees = defaultdict(list)
        for hom_deg_curr in sorted(projectives_curr.keys()):
            for hom_deg_next in projective_next.keys():
                hom_degrees[hom_deg_curr + hom_deg_next].append(
                    (hom_deg_curr, hom_deg_next)
                )

        return hom_degrees

    def projectives_and_thimbles_in_product(
        self,
        number_of_moving_strands,
        hom_degrees,
        projectives_curr,
        projective_next,
        thimbles_current,
        thimbles_next,
    ):
        klrw_algebra = self.klrw_algebra[number_of_moving_strands]
        vv_crossing_degree = klrw_algebra.grading_group.crossing_grading(self.V, self.V)
        thimbles = {}
        projectives_subdivided = {}
        for hom_deg in hom_degrees:
            thimbles[hom_deg] = []
            projectives_subdivided[hom_deg] = defaultdict(list)
            projs_in_hom_deg = projectives_subdivided[hom_deg]
            for hom_deg_curr, hom_deg_next in hom_degrees[hom_deg]:
                for i in range(len(projectives_curr[hom_deg_curr])):
                    thim_curr = thimbles_current[hom_deg_curr][i]
                    proj_curr = projectives_curr[hom_deg_curr][i]
                    for j in range(len(projective_next[hom_deg_next])):
                        thim_next = thimbles_next[hom_deg_next][j]
                        proj_next = projective_next[hom_deg_next][j]
                        new_thimble = ProductThimbles(
                            colored_state=thim_curr.colored_state
                            + thim_next.colored_state,
                            points=thim_curr.points + thim_next.points,
                        )
                        thimbles[hom_deg].append(new_thimble)

                        state_list = []
                        for i in range(self.n + 1):
                            strands_in_segment = new_thimble.colored_state.count(i)
                            state_list = state_list + [self.V] * strands_in_segment

                            # if not the last segment, add separation
                            if i != self.n:
                                state_list = state_list + [self.W]

                        state = klrw_algebra.state(state_list)

                        equ_deg = (
                            proj_curr.equivariant_degree + proj_next.equivariant_degree
                        )
                        # need to take into account contributions
                        # from terms log(1-y_i/y_j)
                        for pt_curr in thim_curr.points:
                            for pt_next in thim_next.points:
                                equ_deg += (
                                    self.pairwise_equ_deg_halved[pt_curr, pt_next]
                                    * vv_crossing_degree
                                )

                        new_proj = KLRWProjectiveModule(state, equ_deg)

                        projs_in_hom_deg[hom_deg_curr, hom_deg_next].append(new_proj)

        return thimbles, projectives_subdivided

    def differential_and_thimbles(
        self, *brane_indices, parallel_processes=1, indices_ordered=False
    ):
        assert brane_indices, "There has to be at least one E-brane."

        if not indices_ordered:
            self.order_intersections()

        if len(brane_indices) == 1:
            # the differential for the zeroth brane
            print("====Making the complex for {} E-brane====".format(brane_indices[0]))
            return self.one_dimensional_differential_projectives_thimbles(
                brane_indices[0]
            )

        diff_curr, projectives_curr, thimbles_curr = self.differential_and_thimbles(
            *brane_indices[:-1],
            parallel_processes=parallel_processes,
            indices_ordered=True,
        )
        diff_next, projectives_next, thimbles_next = self.differential_and_thimbles(
            *brane_indices[-1:],
            parallel_processes=parallel_processes,
            indices_ordered=True,
        )

        print(
            "====Making the complex for "
            + ", ".join(str(i) for i in brane_indices)
            + " E-branes===="
        )

        hom_degrees = self.hom_degrees_in_product(projectives_curr, projectives_next)

        thimbles, projectives_subdivided = self.projectives_and_thimbles_in_product(
            len(brane_indices),
            hom_degrees,
            projectives_curr,
            projectives_next,
            thimbles_curr,
            thimbles_next,
        )

        klrw_algebra = self.klrw_algebra[len(brane_indices)]
        differential = {}
        for hom_deg in hom_degrees:
            if hom_deg + self.degree in hom_degrees:
                left_projectives = projectives_subdivided[hom_deg]
                right_projectives = projectives_subdivided[hom_deg + self.degree]
                matrix_blocks = {}

                for hom_deg_curr, hom_deg_next in hom_degrees[hom_deg]:
                    row_multiindex = hom_deg_curr, hom_deg_next
                    if hom_deg_curr + self.degree in projectives_curr:
                        column_multiindex = hom_deg_curr + self.degree, hom_deg_next
                        block = d_times_one_piece(
                            klrw_algebra,
                            d_curr=diff_curr[hom_deg_curr],
                            curr_right_thimbles=thimbles_curr[
                                hom_deg_curr + self.degree
                            ],
                            curr_left_thimbles=thimbles_curr[hom_deg_curr],
                            next_thimbles=thimbles_next[hom_deg_next],
                            right_projectives=right_projectives[column_multiindex],
                            left_projectives=left_projectives[row_multiindex],
                            hom_add_one_more_strand=self.hom_add_one_more_strand[
                                len(brane_indices) - 1
                            ],
                        )
                        matrix_blocks[row_multiindex, column_multiindex] = block

                    if hom_deg_next + self.degree in projectives_next:
                        column_multiindex = hom_deg_curr, hom_deg_next + self.degree
                        if hom_deg_curr % 2 == 0:
                            sign = ZZ(1)
                        else:
                            sign = -ZZ(1)
                        block = one_times_d_piece(
                            klrw_algebra,
                            d_next=diff_next[hom_deg_next],
                            curr_thimbles=thimbles_curr[hom_deg_curr],
                            next_right_thimbles=thimbles_next[
                                hom_deg_next + self.degree
                            ],
                            next_left_thimbles=thimbles_next[hom_deg_next],
                            right_projectives=right_projectives[column_multiindex],
                            left_projectives=left_projectives[row_multiindex],
                            hom_one_to_many_dots=self.hom_one_to_many_dots[
                                len(brane_indices) - 1
                            ],
                            sign=sign,
                        )
                        matrix_blocks[row_multiindex, column_multiindex] = block

                differential[hom_deg] = CSC_from_dict_of_blocks(matrix_blocks)

        projectives = defaultdict(list)
        for hom_deg, dict_of_lists in projectives_subdivided.items():
            for key in sorted(dict_of_lists.keys()):
                projectives[hom_deg].extend(dict_of_lists[key])

        diff_csc = corrected_diffirential_csc(
            klrw_algebra,
            differential,
            projectives,
            parallel_processes=parallel_processes,
            degree=self.degree,
        )

        return diff_csc, projectives, thimbles

    def complex(
        self,
        *brane_indices,
        pickle="",
        link_name="link",
        folder_path="./",
        parallel_processes=1,
    ):
        """
        Create a complex.

        If pickle=="save", the pieces of data for complex are pickled.
        If pickle=="load", the pieces of data for complex are uppickled from files.
        """
        if pickle == "load":
            from pickle import load

            with open(folder_path + link_name + "_differentials.pickle", "rb") as f:
                differential_csc = load(file=f)
            with open(folder_path + link_name + "_projectives.pickle", "rb") as f:
                projectives = load(file=f)

        else:
            differential_csc, projectives, thimbles = self.differential_and_thimbles(
                *brane_indices,
                parallel_processes=parallel_processes,
                indices_ordered=False,
            )

            if pickle == "save":
                from pickle import dump

                with open(folder_path + link_name + "_differentials.pickle", "wb") as f:
                    dump(differential_csc, file=f)
                with open(folder_path + link_name + "_projectives.pickle", "wb") as f:
                    dump(projectives, file=f)
                with open(folder_path + link_name + "_thimbles.pickle", "wb") as f:
                    dump(thimbles, file=f)

        differential = {
            hom_deg: matrix(
                self.klrw_algebra[len(brane_indices)],
                mat._number_of_rows(),
                len(mat._indptrs()) - 1,
                mat.dict(),
            )
            for hom_deg, mat in differential_csc.items()
        }

        return KLRWPerfectComplex(
            self.klrw_algebra[len(brane_indices)],
            differential,
            projectives,
            degree=self.degree,
        )

    @staticmethod
    def find_homology(
        complex: KLRWPerfectComplex,
        relevant_state: KLRWstate,
        R: PrincipalIdealDomain,
        hom_deg_shift=0,
        equ_deg_shift=0,
        dualize_complex=False,
    ):

        rhom_complex = complex.rhom_to_simple(
            relevant_state, dualize_complex=dualize_complex
        )

        G = rhom_complex.grading_group()
        shift_in_degrees = rhom_complex.grading_group()((hom_deg_shift, equ_deg_shift))
        # transformation to match standard grading in Khovanov homology
        transformation = G.hom(
            [
                G((-1, 0)),
                G((-1, -1)),
            ]
        )

        homology_raw = rhom_complex.homology(base_ring=R)

        homology = {}
        for grading, homology_group in homology_raw.items():
            if homology_group.ngens() != 0:
                homology[transformation(grading) + shift_in_degrees] = homology_group

        if not R.is_field():
            return homology

        lau_poly = LaurentPolynomialRing(ZZ, 2, ["t", "q"])
        t = lau_poly("t")
        q = lau_poly("q")
        poincare_polynomial = sum(
            homology_group.ngens() * t ** grading[0] * q ** grading[1]
            for grading, homology_group in homology.items()
        )

        return poincare_polynomial

    def link_homology(
        self,
        R: PrincipalIdealDomain,
        hom_deg_shift=0,
        equ_deg_shift=0,
        dualize_complex=False,
        pickle="",
        link_name="link",
        folder_path="./",
        parallel_processes=1,
    ):
        """
        Working over R that is a PID.
        [works on fields and integers, modules over
        other PIDs not fully implemented in Sage yet]
        Returns a Poincare Polynomial if R is a field
        and a dictionary {degrees:invariant factors}
        if R is the ring of integers
        """

        complex = self.complex(
            *range(len(self.branes)),
            pickle=pickle,
            link_name=link_name,
            folder_path=folder_path,
            parallel_processes=parallel_processes,
        )

        relevant_state = self.klrw_algebra[len(self.branes)].state(
            [self.W, self.V, self.W] * self.k + [self.W] * (self.n - 2 * self.k)
        )

        return self.find_homology(
            complex=complex,
            relevant_state=relevant_state,
            R=R,
            hom_deg_shift=hom_deg_shift,
            equ_deg_shift=equ_deg_shift,
            dualize_complex=dualize_complex,
        )
