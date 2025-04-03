from dataclasses import dataclass, replace
from collections import defaultdict
from types import MappingProxyType
from itertools import product
from typing import Iterable
import operator

from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.polynomial.multi_polynomial_libsingular import (
    MPolynomialRing_libsingular as PolynomialRing_sing,
)
from sage.rings.polynomial.laurent_polynomial_ring import (
    LaurentPolynomialRing_mpair as LaurentPolynomialRing,
)
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing

# from sage.rings.polynomial.laurent_polynomial import LaurentPolynomial_mpair
from sage.rings.polynomial.polydict import ETuple
from sage.rings.ring import CommutativeRing
from sage.categories.action import Action

from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.misc.misc_c import prod

from klrw.framed_dynkin import (
    NodeInFramedQuiver,
    FramedDynkinDiagram_with_dimensions,
)
from klrw.gradings import (
    QuiverGradingGroup,
    QuiverGradingSelfCrossingLabel,
    QuiverGradingUnorientedCrossingLabel,
    QuiverGradingDotLabel,
    QuiverGradingEquvariantLabel,
)


@dataclass(slots=True)
class QuiverParameter:
    # index : object
    name: str
    position: int
    monomial: object
    invertible: bool = True

    def __repr__(self):
        return self.monomial.__repr__()


class VariableIndex:
    pass


@dataclass(frozen=True)
class DotVariableIndex(VariableIndex):
    vertex: NodeInFramedQuiver
    number: int | None

    def __repr__(self):
        return "Dot index {} #{}".format(self.vertex, self.number)


@dataclass(frozen=True)
class VertexVariableIndex(VariableIndex):
    vertex: NodeInFramedQuiver

    def __repr__(self):
        return "Vertex index {}".format(self.vertex)


@dataclass(frozen=True)
class EdgeVariableIndex(VariableIndex):
    vertex_tail: NodeInFramedQuiver
    vertex_head: NodeInFramedQuiver

    def __repr__(self):
        return "Edge index {}->{}".format(self.vertex_tail, self.vertex_head)


def exps_for_dots_of_degree(etuples_degrees: MappingProxyType | tuple, degree):
    if isinstance(etuples_degrees, MappingProxyType):
        etuples_degrees = tuple(x for x in etuples_degrees.items())
    et, deg = etuples_degrees[0]
    if len(etuples_degrees) == 1:
        if int(degree) % int(deg) == 0:
            yield et.emul(int(degree) // int(deg))
        return
    else:
        power_of_first = int(degree) // int(deg)
        for power in range(power_of_first + 1):
            for rec_et in exps_for_dots_of_degree(
                etuples_degrees[1:],
                degree - power * deg,
            ):
                yield rec_et.eadd_scaled(et, power)


def exps_for_dots_by_number(etuples: Iterable, number):
    """
    Returns all exponent ETuples with desired number of dots.

    Similar to `exps_for_dots_of_degree`, but weight of every dot is 1.
    """
    if int(number) < 0:
        return

    etuples = tuple(etuples)

    et = etuples[0]
    if len(etuples) == 1:
        if int(number) >= 0:
            yield et.emul(int(number))
        return
    else:
        power_of_first = int(number)
        for power in range(power_of_first + 1):
            for rec_et in exps_for_dots_by_number(
                etuples[1:],
                power_of_first - power,
            ):
                yield rec_et.eadd_scaled(et, power)


# TODO: add an abstract class KLRWDotsAlgebra(CoomutativeRing, UniqueRep),
# inherit upstairs and downstairs versions
class KLRWDotsAlgebra(UniqueRepresentation, CommutativeRing):
    # @weak_cached_function(cache=128)  # automatically a staticmethod
    @staticmethod
    def __classcall_private__(cls, *args, invertible_parameters=False, **kwargs):
        if not invertible_parameters:
            # TODO: add the case when the list of invertible parameters is empty
            return KLRWUpstairsDotsAlgebra(
                invertible_parameters=invertible_parameters, *args, **kwargs
            )
        else:
            return KLRWUpstairsDotsAlgebra_invertible_parameters(
                invertible_parameters=invertible_parameters, *args, **kwargs
            )

    def __init__(
        self,
        base_ring,
        quiver_data: FramedDynkinDiagram_with_dimensions,
        no_deformations=True,
        default_vertex_parameter=None,
        default_edge_parameter=None,
        invertible_parameters=False,
        order="degrevlex",
        **prefixes,
    ):
        # remember initialization data for pickle/unpickle
        self.quiver_data = quiver_data.immutable_copy()
        self.no_deformations = no_deformations
        self.default_vertex_parameter = default_vertex_parameter
        self.default_edge_parameter = default_edge_parameter
        self.invertible_parameters = invertible_parameters
        self.prefixes = prefixes

        no_vertex_parameters = default_vertex_parameter is not None
        no_edge_parameters = default_edge_parameter is not None
        nondot_names, dot_names = quiver_data.names(
            no_deformations=no_deformations,
            no_vertex_parameters=no_vertex_parameters,
            no_edge_parameters=no_edge_parameters,
            **prefixes,
        )

        nondot_names_list = [name for _, name in nondot_names.items()]
        dot_names_list = [name for _, name in dot_names.items()]
        self.init_algebra(base_ring, nondot_names_list, dot_names_list, order)

        # type of variables is uniquely determined by its index data
        self.variables = {
            index: QuiverParameter(
                name,
                position,
                self(name),
                invertible_parameters,
            )
            for position, (index, name) in enumerate(nondot_names.items())
        }
        self.variables |= {
            index: QuiverParameter(
                name,
                position + len(self.variables),  # dots follow non-dots
                self(name),
                False,  # dots are not invertible
            )
            for position, (index, name) in enumerate(dot_names.items())
        }

        # adds default values to self.variables
        self.assign_default_values(
            quiver_data,
            default_vertex_parameter=default_vertex_parameter,
            default_edge_parameter=default_edge_parameter,
            no_deformations=no_deformations,
        )

        # make immutable
        self.variables = MappingProxyType(self.variables)

    def __reduce__(self):
        from functools import partial

        return (
            partial(self.__class__, **self.prefixes),
            (
                self.base(),
                self.quiver_data,
                self.no_deformations,
                self.default_vertex_parameter,
                self.default_edge_parameter,
                self.invertible_parameters,
                self.term_order().name(),
            ),
        )

    @lazy_attribute
    def without_dots(self):
        """
        Return the subalgebra algebra with no dots.
        """
        quiver_data = self.quiver_data.with_zero_dimensions(self.quiver_data.quiver)
        return KLRWDotsAlgebra(
            base_ring=self.base(),
            quiver_data=quiver_data,
            no_deformations=self.no_deformations,
            default_vertex_parameter=self.default_vertex_parameter,
            default_edge_parameter=self.default_edge_parameter,
            term_order=self.term_order().name(),
            invertible_parameters=self.invertible_parameters,
            **self.prefixes,
        )

    def assign_default_values(
        self,
        quiver_data,
        no_deformations=True,
        default_vertex_parameter=None,
        default_edge_parameter=None,
    ):
        # TODO: add default parameters' options,
        # maybe by changing self.variables to a custom dictionary
        # second set of framing variables rescaled to 1
        quiver = quiver_data.quiver
        if default_edge_parameter is None:
            for n1, n2 in product(quiver.vertices(), quiver.vertices()):
                index = EdgeVariableIndex(n1, n2)
                if index not in self.variables:
                    self.variables[index] = QuiverParameter(
                        None, None, self.one(), True
                    )
        else:
            default_value = self(default_edge_parameter)
            is_unit = default_value.is_unit()
            for n1, n2 in product(quiver.vertices(), quiver.vertices()):
                index = EdgeVariableIndex(n1, n2)
                self.variables[index] = QuiverParameter(
                    None, None, default_value, is_unit
                )
        if default_vertex_parameter is not None:
            default_value = self(default_vertex_parameter)
            is_unit = default_value.is_unit()
            for key in quiver.non_framing_nodes():
                index = VertexVariableIndex(key)
                self.variables[index] = QuiverParameter(
                    None, None, default_value, is_unit
                )
        if no_deformations:
            for key in quiver_data.KLRW_deformations_names():
                self.variables[key] = QuiverParameter(None, None, self.zero(), False)

    @lazy_attribute
    def index_by_position(self):
        result = {
            var.position: index
            for index, var in self.variables.items()
            if var.position is not None
        }
        result = MappingProxyType(result)
        return result

    def center_gens(self):
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    continue
            if variable.name is not None:
                yield variable.monomial
        yield from self.symmetric_dots_gens()

    def basis_modulo_symmetric_dots(self):
        variables = []
        degree_ranges = []
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    degree_ranges.append(range(index.number))
                    variables.append(variable.monomial)
        degree_iterator = product(*degree_ranges)
        for degree in degree_iterator:
            yield prod(var**deg for var, deg in zip(variables, degree))

    @lazy_attribute
    def number_of_non_dot_params(self):
        return sum(
            1
            for index, var in self.variables.items()
            if (not isinstance(index, DotVariableIndex) and var.position is not None)
        )

    @cached_method
    def dot_exptuples_and_degrees(self, grading_group: QuiverGradingGroup):
        result = {}
        relevant_grading_index = grading_group.equivariant_grading_label
        for index, var in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    exps = var.monomial.exponents()
                    # must be a monomial
                    assert len(exps) == 1, repr(var)
                    grading = grading_group.dot_algebra_grading(index)
                    result[exps[0]] = grading.coefficient(relevant_grading_index)
        return MappingProxyType(result)

    @lazy_attribute
    def dot_exptuples_by_color(self):
        result = defaultdict(list)
        for index, var in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    exps = var.monomial.exponents()
                    # must be a monomial
                    assert len(exps) == 1, repr(var)
                    color = index.vertex
                    result[color] += [exps[0]]
        return MappingProxyType(result)

    @cached_method
    def degrees_by_position(self, grading_group: QuiverGradingGroup):
        result = [list() for _ in range(self.ngens())]
        for index, var in self.variables.items():
            if var.position is not None:
                result[var.position] = grading_group.dot_algebra_grading(index)

        return tuple(result)

    @cached_method
    def degrees_by_position_ignoring_dots(self, grading_group: QuiverGradingGroup):
        result = [list() for _ in range(self.ngens())]
        for index, var in self.variables.items():
            if var.position is not None:
                if isinstance(index, DotVariableIndex):
                    # zero if a dot
                    result[var.position] = grading_group()
                else:
                    result[var.position] = grading_group.dot_algebra_grading(index)

        return tuple(result)

    @lazy_attribute
    def dot_variables(self):
        result = defaultdict(list)
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                result[index.vertex].append(variable)

        # make everything immutable
        for key in result:
            result[key] = tuple(result[key])

        return MappingProxyType(result)

    def symmetric_dots_gens(self):
        elementary_symmetric_polynomials = {}
        for index, variable in self.variables.items():
            if isinstance(index, DotVariableIndex):
                if not index.vertex.is_framing():
                    if index.vertex in elementary_symmetric_polynomials:
                        e_list = elementary_symmetric_polynomials[index.vertex]
                        # add the top elementary symmetric polynomial
                        e_list.append(variable.monomial * e_list[-1])
                        # modify intermediate symmetric polynomials
                        for i in range(len(e_list) - 2, 0, -1):
                            e_list[i] += variable.monomial * e_list[i - 1]
                        # modify the first elementary symmetric polynomial
                        e_list[0] += variable.monomial
                    else:
                        # for one variable there is only
                        # the first elementary symmetric polynomial
                        elementary_symmetric_polynomials[index.vertex] = [
                            variable.monomial
                        ]

        for _, e_list in elementary_symmetric_polynomials.items():
            yield from e_list

    def basis_by_degree(self, degree):
        return tuple(self.monomial(*et) for et in self.exps_by_degree(degree))

    @cached_method
    def exps_by_degree(self, degree):
        """
        Return a tuple of exponents of a given degree.

        Monomials of these degrees form a basis over
        the subalgebra of elements of zero degree.
        """

        # we make a polynomial with monomials representing all
        # possible monomials of given degree
        result = self.one()
        # first we take into account "easy" degrees
        # where powers of varibles are uniquely reconstructed
        grading_group = degree.parent()
        degrees_taken_into_account = grading_group.zero()
        for label, coeff in degree:
            if isinstance(label, QuiverGradingSelfCrossingLabel):
                index = VertexVariableIndex(label.vertex)
                variable = self.variables[index]
                if coeff < 0:
                    if not variable.invertible:
                        # we prohibit inverse powers
                        return tuple()
                degrees_taken_into_account += coeff * grading_group.dot_algebra_grading(
                    index
                )
                result *= variable.monomial**coeff

            elif isinstance(label, QuiverGradingUnorientedCrossingLabel):
                if int(coeff) % int(2) != 0:
                    return tuple()
                d = int(coeff) // int(2)
                v1, v2 = label.edge
                edge_index = EdgeVariableIndex(v1, v2)
                edge_opp_index = EdgeVariableIndex(v2, v1)
                if v2.is_framing():
                    variable = self.variables[edge_index]
                    degree_part = grading_group.dot_algebra_grading(edge_index)
                elif v1.is_framing():
                    variable = self.variables[edge_opp_index]
                    degree_part = grading_group.dot_algebra_grading(edge_opp_index)
                elif not grading_group.dot_scaling:
                    # in this case t_ij and t_ji have the same grading,
                    # t_ij/t_ji is in the subalgebra of degree zero elements
                    variable = self.variables[edge_index]
                    degree_part = grading_group.dot_algebra_grading(edge_index)
                else:
                    # in other cases this degree has to be treated later by an ILP.
                    continue

                if d < 0:
                    if not variable.invertible:
                        return tuple()
                result *= variable.monomial**d
                degrees_taken_into_account += d * degree_part

            elif label != grading_group.equivariant_grading_label and not isinstance(
                label, QuiverGradingDotLabel
            ):
                raise ValueError("Unknown type of grading: {}".format(label))

        # take care of dots if they are only constrained by ordinary equivariant degree
        if not grading_group.dot_scaling:
            excess_degree = degree - degrees_taken_into_account
            coeff = excess_degree.coefficient(grading_group.equivariant_grading_label)
            if int(coeff) % int(2) != 0 or coeff < 0:
                return tuple()
            new_part = sum(
                self.monomial(*etuple)
                for etuple in exps_for_dots_of_degree(
                    self.dot_exptuples_and_degrees(grading_group),
                    coeff,
                )
            )
            degrees_taken_into_account += grading_group.term(
                grading_group.equivariant_grading_label,
                coeff,
            )
            result *= new_part

        excess_degree = degree - degrees_taken_into_account
        if not excess_degree.is_zero():
            excess_part = self.zero()
            indices_inv, indices_non_inv = self._indices_for_ilp_on_gradings_()
            indices = indices_inv + indices_non_inv
            solutions = self._solve_ilp_on_gradings_(excess_degree)
            if solutions.shape[0] == 0:
                return tuple()
            for i in range(solutions.shape[0]):
                part_from_a_solution = self.one()
                for j, index in enumerate(indices):
                    if isinstance(index, EdgeVariableIndex):
                        part_from_a_solution *= (
                            self.variables[index].monomial ** solutions[i, j]
                        )
                    elif isinstance(index, DotVariableIndex):
                        part_from_a_solution *= (
                            self.complete_homogeneous_symmetric_in_dots(
                                vertex_of_dots=index.vertex, n=solutions[i, j]
                            )
                        )
                    else:
                        raise ValueError("Unxepected variable index: {}".format(index))
                excess_part += part_from_a_solution
            result *= excess_part

        return tuple(x for x in result.exponents())

    @cached_method
    def exps_by_degree_and_parameter_part(self, degree, parameter_part: ETuple):
        """
        Return a tuple of exponents of a given degree and parameter part.

        Monomials differ by the number of dots.
        """

        # we make a polynomial with monomials representing all
        # possible monomials of given degree
        result = self.monomial(*parameter_part)
        # first we take into account "easy" degrees
        # where powers of varibles are uniquely reconstructed
        grading_group = degree.parent()
        total_degree_of_new_parts = grading_group.zero()
        excess_degree = degree - self.exp_degree(parameter_part, grading_group)
        for label, coeff in excess_degree:
            # only two types of gradings are allowed in for dots
            if not isinstance(
                label, QuiverGradingDotLabel | QuiverGradingEquvariantLabel
            ):
                return tuple()

            if isinstance(label, QuiverGradingEquvariantLabel):
                # Purely equivariant gradings do not matter.
                # We will check that it's correct later.
                continue

            color = label.vertex
            new_part = sum(
                self.monomial(*etuple)
                for etuple in exps_for_dots_by_number(
                    self.dot_exptuples_by_color[color],
                    coeff,
                )
            )

            if not new_part:
                return tuple()

            total_degree_of_new_parts += coeff * grading_group.dot_algebra_grading(
                DotVariableIndex(color, None)
            )

            result *= new_part

        # checking if purely equivariant part matches
        if total_degree_of_new_parts == excess_degree:
            return tuple(x for x in result.exponents())
        else:
            return tuple()

    @cached_method
    def _indices_for_ilp_on_gradings_(self):
        relevant_invertible_indices = set()
        relevant_not_invertible_indices = set()
        for index, var in self.variables.items():
            if var.name is not None:
                if isinstance(index, DotVariableIndex):
                    dot_index = replace(index, number=1)
                    relevant_not_invertible_indices.add(dot_index)
                if isinstance(index, EdgeVariableIndex):
                    if (
                        not index.vertex_head.is_framing()
                        and not index.vertex_tail.is_framing()
                    ):
                        if var.invertible:
                            relevant_invertible_indices.add(index)
                        else:
                            relevant_not_invertible_indices.add(index)

        relevant_invertible_indices = tuple(relevant_invertible_indices)
        relevant_not_invertible_indices = tuple(relevant_not_invertible_indices)

        return relevant_invertible_indices, relevant_not_invertible_indices

    @cached_method
    def _ilp_on_gradings_(self, grading_group):
        from scipy.sparse import dok_array
        from numpy import intc

        relevant_invertible_indices, relevant_not_invertible_indices = (
            self._indices_for_ilp_on_gradings_()
        )

        A_inv = dok_array(
            (
                grading_group.dimension(),
                len(relevant_invertible_indices),
            ),
            dtype=intc,
        )
        for j in range(len(relevant_invertible_indices)):
            index = relevant_invertible_indices[j]
            monomial = self.variables[index].monomial
            degree = self.element_degree(monomial, grading_group)
            for i, v in degree._vector_(sparse=True).dict(copy=False).items():
                A_inv[i, j] = v

        A_non_inv = dok_array(
            (
                grading_group.dimension(),
                len(relevant_not_invertible_indices),
            ),
            dtype=intc,
        )
        for j in range(len(relevant_not_invertible_indices)):
            index = relevant_not_invertible_indices[j]
            monomial = self.variables[index].monomial
            degree = self.element_degree(monomial, grading_group)
            for i, v in degree._vector_(sparse=True).dict(copy=False).items():
                A_non_inv[i, j] = v

        return A_inv.tocsr(), A_non_inv.tocsr()

    @cached_method
    def _solve_ilp_on_gradings_(self, degree):
        from scipy.sparse import dok_array
        from numpy import intc, zeros
        import gurobipy as gp

        grading_group = degree.parent()

        A_inv, A_non_inv = self._ilp_on_gradings_(grading_group)

        # a bug in Gurobi since 12.0.1:
        # subtraction is not supported with scipy's sparse matrices,
        # but addition is supported.
        # So, we use addition instead of == that does subtraction.
        y_transposed_neg = dok_array((1, grading_group.dimension()), dtype=intc)
        for i, v in degree._vector_(sparse=True).dict(copy=False).items():
            y_transposed_neg[0, i] = -v

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as m:
                m.Params.PoolSearchMode = 2
                m.Params.PoolSolutions = gp.GRB.MAXINT

                x_non_inv = m.addMVar(A_non_inv.shape[1], lb=int(0), vtype="I")
                x_inv = m.addMVar(
                    A_inv.shape[1],
                    lb=-float("inf"),
                    vtype="I",
                )

                if A_inv.shape[1] == 0:
                    m.addConstr(A_non_inv @ x_non_inv + y_transposed_neg == 0)
                elif A_non_inv.shape[1] == 0:
                    m.addConstr(A_inv @ x_inv + y_transposed_neg == 0)
                else:
                    m.addConstr(
                        A_inv @ x_inv + A_non_inv @ x_non_inv + y_transposed_neg == 0
                    )

                m.optimize()
                # if the system is inconsistent
                if m.Status == 3:
                    return zeros(
                        shape=(0, x_inv.size + x_non_inv.size),
                        dtype=intc,
                    )

                n_solutions = m.SolCount

                solutions = zeros(
                    shape=(n_solutions, x_inv.size + x_non_inv.size),
                    dtype=intc,
                )
                for n in range(n_solutions):
                    m.Params.SolutionNumber = n
                    solutions[n, : x_inv.size] = x_inv.Xn
                    solutions[n, x_inv.size :] = x_non_inv.Xn

        return solutions

    def exp_degree(
        self,
        exp: ETuple,
        grading_group: QuiverGradingGroup,
        ignoring_dots=False,
    ):
        if ignoring_dots:
            degrees_by_position = self.degrees_by_position_ignoring_dots(grading_group)
        else:
            degrees_by_position = self.degrees_by_position(grading_group)
        return grading_group.linear_combination(
            (degrees_by_position[position], power)
            for position, power in exp.sparse_iter()
        )

    # libsingular wrapper in Sage does not allow to change elements for a wrapper,
    # so we define degree in the parent
    def element_degree(
        self,
        element,
        grading_group: QuiverGradingGroup,
        check_if_homogeneous=False,
        ignoring_dots=False,
    ):
        if ignoring_dots:
            degrees_by_position = self.degrees_by_position_ignoring_dots(grading_group)
        else:
            degrees_by_position = self.degrees_by_position(grading_group)
        # zero elements return None as degree
        degree = None
        for exp, scalar in element.iterator_exp_coeff():
            if not scalar.is_zero():
                term_degree = grading_group.linear_combination(
                    (degrees_by_position[position], power)
                    for position, power in exp.sparse_iter()
                )

                if not check_if_homogeneous:
                    return term_degree
                elif degree is None:
                    degree = term_degree
                elif degree != term_degree:
                    raise ValueError("The dots are not homogeneous! {}".format(element))

        return degree

    @cached_method
    def no_parameters_of_zero_degree(self, grading_group: QuiverGradingGroup):
        return all(not deg.is_zero() for deg in self.degrees_by_position(grading_group))

    def is_element_symmetric(self, element):
        for v in self.quiver_data.non_framing_nodes():
            dim_v = self.quiver_data[v]
            for i in range(1, dim_v):
                x_i = self.dot_variable(v, i)
                x_i_1 = self.dot_variable(v, i + 1)
                swap = {
                    x_i.name: x_i_1.monomial,
                    x_i_1.name: x_i.monomial,
                }
                swapped_element = element.subs(**swap)
                if swapped_element != element:
                    return False
        return True

    def etuple_ignoring_dots(self, et: ETuple):
        # we assume that the non-zero dots have first positions
        # This is not fast, similar to slicing of ETuples
        data = {
            ind: exp
            for ind, exp in et.sparse_iter()
            if ind < self.number_of_non_dot_params
        }
        return ETuple(data=data, length=self.ngens())

    def monomial_quotient_by_dots(self, et: ETuple):
        """
        Returns a monomial in the quotient by dots.
        (can be zero)
        """
        if self.etuple_has_dots(et):
            return self.without_dots.zero()
        new_etuple = self.without_dots.etuple_ignoring_dots(et)
        return self.without_dots.monomial(*new_etuple)

    def etuple_has_dots(self, et: ETuple):
        # we assume that the non-zero dots have first positions
        return any(ind >= self.number_of_non_dot_params for ind, _ in et.sparse_iter())

    def geometric_part(self, element):
        return sum(
            coeff * self.monomial(*exp)
            for exp, coeff in element.iterator_exp_coeff()
            if self.etuple_ignoring_dots(exp).is_constant()
        )

    def hom_from_dots_map(self, codomain, map: MappingProxyType):
        variables_images = [
            (
                codomain.variables[map[index]].monomial
                if index in map
                else codomain.variables[index].monomial
            )
            for index in self.variables
            if self.variables[index].name is not None
        ]

        return self.hom(variables_images, codomain)

    def _coerce_map_from_(self, other):
        if isinstance(other, KLRWDotsAlgebra):
            # we allow coercion if both quivers are equal
            # and dimensions are equal
            compatible_quiver_data = self.quiver_data == other.quiver_data
            # or if we coerce from the ring with no dots
            compatible_quiver_data |= (
                self.quiver_data.quiver == other.quiver_data.quiver
                and not other.quiver_data.dimensions(copy=False)
            )
            if compatible_quiver_data:
                can_coerce = self.base().has_coerce_map_from(other.base())
                if other.default_vertex_parameter is not None:
                    can_coerce &= (
                        self.default_vertex_parameter == other.default_vertex_parameter
                    )
                if other.default_edge_parameter is not None:
                    can_coerce &= (
                        self.default_edge_parameter == other.default_edge_parameter
                    )
                if can_coerce:
                    variables_images = [None] * other.ngens()
                    for index, var in other.variables.items():
                        if var.name is not None:
                            var_image = self.variables[index].monomial
                            variables_images[var.position] = var_image
                    return other.hom(tuple(variables_images), self)
            return

        return super()._coerce_map_from_(other)

    def quotient_by_dots(self, elem):
        """
        Kill dots in `elem`.

        Gets elements in the pure paramenter ring.
        """
        return sum(
            (
                coeff*self.monomial_quotient_by_dots(exp)
                for exp, coeff in elem.iterator_exp_coeff()
            ),
            start=self.without_dots.zero(),
        )

    def hom_to_simple(self, elem):
        """
        Kill dots in `elem`, set all multiplicative parameters to 1.

        Gets scalars in the base ring.
        """
        return sum(
            (
                coeff
                for exp, coeff in elem.iterator_exp_coeff()
                if not self.etuple_has_dots(exp)
            ),
            start=self.base().zero(),
        )

    def variable(self, *index):
        if len(index) == 1:
            assert isinstance(
                index[0], NodeInFramedQuiver
            ), "Unknown index type, {}".format(index)
            return self.vertex_variable(index[0])

        assert len(index) == 2, "Unknown index type, {}".format(index)

        if isinstance(index[1], NodeInFramedQuiver):
            return self.edge_variable(*index)
        else:
            return self.dot_variable(*index)

    def dot_variable(self, vertex, number):
        return self.variables[DotVariableIndex(vertex=vertex, number=number)]

    def vertex_variable(self, vertex):
        return self.variables[VertexVariableIndex(vertex=vertex)]

    def edge_variable(self, vertex_tail, vertex_head):
        return self.variables[
            EdgeVariableIndex(vertex_tail=vertex_tail, vertex_head=vertex_head)
        ]

    @cached_method
    def complete_homogeneous_symmetric_in_dots(
        self, vertex_of_dots: NodeInFramedQuiver, n: int
    ):
        from sage.combinat.sf.sf import SymmetricFunctions

        h_n = SymmetricFunctions(self.base()).complete()([n])
        n_vars = self.quiver_data[vertex_of_dots]
        h_n = h_n.expand(n=n_vars)
        images = [self.dot_variable(vertex_of_dots, i + 1) for i in range(n_vars)]
        m = h_n.parent().hom(images, self)
        return m(h_n)

    def _dict_of_dots_iterator_(self, element):
        """
        Iterates over monomials.

        Yields a pairs
        `(coeff, dic)`
        where `coeff` is a polynomial in non-dots
        and `dic` is a dictionary that keeps the data about
        dots in form
        `{dot_index : power}`
        Each type of `dic` appears at most once.
        """
        # by default each entry is zero in `self.without_dots`
        result = defaultdict(self.without_dots)
        for exp, scalar in element.iterator_exp_coeff():
            coeff_dict = {}
            dots_dict = {}

            # we assume that the non-zero dots have first positions
            for i, pow in exp.sparse_iter():
                if i < self.number_of_non_dot_params:
                    coeff_dict[i] = pow
                else:
                    dots_dict[i] = pow

            # could also do
            # `dots_etuple = exp[self.number_of_non_dot_params:]`
            # and
            # `etuple = exp[:self.number_of_non_dot_params]`,
            # but slices can be slow for many zeroes
            # (see implementation of ETuple)
            dots_etuple = ETuple(dots_dict, self.ngens())
            etuple = ETuple(coeff_dict, self.without_dots.ngens())
            coeff = scalar * self.without_dots.monomial(*etuple)
            result[dots_etuple] += coeff

        for dots_etuple, coeff in result.items():
            dots_dict = {
                self.index_by_position[i]: pow for i, pow in dots_etuple.sparse_iter()
            }

            dots_dict = MappingProxyType(dots_dict)
            yield (coeff, dots_dict)

    @staticmethod
    def tensor_product(*terms):
        return TensorProductOfDotAlgebras(*terms)

    def __matmul__(self, other):
        return self.tensor_product(self, other)

    def _get_action_(self, other, op, self_on_left):
        if op == operator.matmul:
            if self_on_left:
                if isinstance(other, KLRWDotsAlgebra):
                    return TensorMultiplication(left_parent=self, right_parent=other)
        return super()._get_action_(other, op, self_on_left)

    @lazy_attribute
    def _self_action(self):
        return TensorMultiplication(left_parent=self, right_parent=self)


class KLRWUpstairsDotsAlgebra(KLRWDotsAlgebra, PolynomialRing_sing):
    class Element(PolynomialRing_sing.Element):
        def _matmul_(self, other):
            """
            Used *only* in the special case when both self and other
            have the same parent.
            Because of a technical details in the coercion model
            this method is called instead of action.
            """
            return self.parent()._self_action.act(self, other)

    def init_algebra(self, base_ring, nondot_names_list, dot_names_list, order):
        super(KLRWDotsAlgebra, self).__init__(
            base_ring,
            len(nondot_names_list) + len(dot_names_list),
            nondot_names_list + dot_names_list,
            order,
        )


class KLRWUpstairsDotsAlgebra_invertible_parameters(
    KLRWDotsAlgebra, LaurentPolynomialRing
):
    class Element(LaurentPolynomialRing.Element):
        def _matmul_(self, other):
            """
            Used *only* in the special case when both self and other
            have the same parent.
            Because of a technical details in the coercion model
            this method is called instead of action.
            """
            return self.parent()._self_action.act(self, other)

    def init_algebra(self, base_ring, nondot_names_list, dot_names_list, order):
        polynomial_ring = PolynomialRing(
            base_ring,
            len(nondot_names_list) + len(dot_names_list),
            nondot_names_list + dot_names_list,
            order=order,
        )
        super(KLRWDotsAlgebra, self).__init__(polynomial_ring)


class TensorProductOfDotAlgebras(KLRWDotsAlgebra):
    """
    Tensor product over the parameter algebra.
    """

    @staticmethod
    def __classcall_private__(
        cls,
        *dot_algebras,
    ):
        from klrw.misc import get_from_all_and_assert_equality

        invertible_parameters = get_from_all_and_assert_equality(
            lambda x: x.invertible_parameters, dot_algebras
        )
        if not invertible_parameters:
            return TensorProductOfKLRWUpstairsDotsAlgebras(dot_algebras)
        else:
            return TensorProductOfKLRWUpstairsDotsAlgebras_invertible_parameters(
                dot_algebras
            )

    def __init__(self, dot_algebras: tuple[KLRWDotsAlgebra]):
        from klrw.misc import get_from_all_and_assert_equality

        assert dot_algebras, "Need at least one dot algebra"
        assert all(isinstance(alg, KLRWDotsAlgebra) for alg in dot_algebras)
        self._parts = dot_algebras
        quiver = get_from_all_and_assert_equality(
            lambda x: x.quiver_data.quiver, dot_algebras
        )
        quiver_data = FramedDynkinDiagram_with_dimensions.with_zero_dimensions(quiver)
        # we make an array where `i`th entry is the quiver with
        # dimensions that are the sum of dimensions of the first `i`
        # dot algebras
        self._partial_quiver_data = [quiver_data.immutable_copy()]
        for alg in dot_algebras:
            for vertex, dim in alg.quiver_data.dimensions(copy=False).items():
                quiver_data[vertex] += dim
            self._partial_quiver_data.append(quiver_data.immutable_copy())

        # TODO: fix _reduction and get all the data from reduction?
        base_ring = get_from_all_and_assert_equality(
            lambda x: x.base_ring(), dot_algebras
        )
        order = get_from_all_and_assert_equality(lambda x: x.term_order(), dot_algebras)
        parameters_names = [
            "no_deformations",
            "default_vertex_parameter",
            "default_edge_parameter",
            "invertible_parameters",
        ]
        parameters = {}
        for name in parameters_names:
            parameters[name] = get_from_all_and_assert_equality(
                lambda x: getattr(x, name), dot_algebras
            )
        prefixes = get_from_all_and_assert_equality(lambda x: x.prefixes, dot_algebras)
        parameters |= prefixes

        super().__init__(
            base_ring=base_ring,
            quiver_data=self._partial_quiver_data[-1],
            order=order,
            **parameters,
        )

    @cached_method
    def embedding(self, i):
        """
        Embeds a piece into the tensor product.

        Dot algebras are unital, so we can send `x` to
        `1 @ ... @ x @ ... @ 1`
        where x is on the `i`th position.
        """
        i = int(i)
        assert i >= 0
        assert i < len(self._parts)
        part_dot_algebra = self._parts[i]
        dims_before_part = self._partial_quiver_data[i]
        variables_images = [None] * part_dot_algebra.ngens()
        for index, var in part_dot_algebra.variables.items():
            if var.position is not None:
                if isinstance(index, DotVariableIndex):
                    vertex = index.vertex
                    new_index = DotVariableIndex(
                        vertex, index.number + dims_before_part[vertex]
                    )
                else:
                    new_index = index
                image = self.variables[new_index].monomial
                variables_images[var.position] = image

        return part_dot_algebra.hom(variables_images, codomain=self)

    @cached_method
    def projection(self, i):
        """
        Projects the tensor product onto a piece.

        Kills all dots that don't belong to the piece.

        Warning: so far does not work for invertible parameters,
        because Sage checks if the images of dots are invertible.
        """
        i = int(i)
        assert i >= 0
        assert i < len(self._parts)
        part_dot_algebra = self._parts[i]
        dims_before_part = self._partial_quiver_data[i]
        dims_after_part = self._partial_quiver_data[i + 1]
        variables_images = [None] * self.ngens()
        for index, var in self.variables.items():
            if var.position is not None:
                if isinstance(index, DotVariableIndex):
                    vertex = index.vertex
                    upper_bound = dims_after_part[vertex]
                    lower_bound = dims_before_part[vertex]
                    if index.number <= upper_bound and index.number > lower_bound:
                        new_index = DotVariableIndex(vertex, index.number - lower_bound)
                    else:
                        new_index = None
                else:
                    new_index = index
                if new_index is not None:
                    image = self.variables[new_index].monomial
                    variables_images[var.position] = image
                else:
                    variables_images[var.position] = part_dot_algebra.zero()

        return self.hom(variables_images, codomain=part_dot_algebra)

    def tensor_product_of_elements(self, *elements):
        """
        Construct an element from parts.
        """
        return prod(self.embedding(i)(element) for i, element in enumerate(elements))


class TensorProductOfKLRWUpstairsDotsAlgebras(
    TensorProductOfDotAlgebras, KLRWUpstairsDotsAlgebra
):
    pass


class TensorProductOfKLRWUpstairsDotsAlgebras_invertible_parameters(
    TensorProductOfDotAlgebras, KLRWUpstairsDotsAlgebra_invertible_parameters
):
    pass


class TensorMultiplication(Action):
    """
    Gives tensor product of two dot algebra elements.

    The solution is similar to matrix multiplication is Sage.
    [see MatrixMatrixAction in MatrixSpace]
    Coercion model in Sage requires both elements in
    the usual product to have the same parent, but in
    actions this is no longer the case. So we use the class Action.
    """

    def __init__(
        self,
        left_parent: KLRWDotsAlgebra,
        right_parent: KLRWDotsAlgebra,
    ):
        Action.__init__(
            self, G=left_parent, S=right_parent, is_left=True, op=operator.matmul
        )
        self._left_domain = self.actor()
        self._right_domain = self.domain()
        self._codomain = self._left_domain @ self._right_domain

    def _act_(self, left, right):
        # return self._codomain.embedding(0)(left) * self._codomain.embedding(1)(right)
        return self._codomain.tensor_product_of_elements(left, right)
