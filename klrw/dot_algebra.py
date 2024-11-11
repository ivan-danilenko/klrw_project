from dataclasses import dataclass, replace
from collections import defaultdict
from types import MappingProxyType
from itertools import product

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

from sage.misc.cachefunc import cached_method, weak_cached_function
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


# TODO: add an abstract class KLRWDotsAlgebra(CoomutativeRing, UniqueRep),
# inherit upstairs and downstairs versions
class KLRWDotsAlgebra(CommutativeRing, UniqueRepresentation):
    @weak_cached_function(cache=128)  # automatically a staticmethod
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
        return sum(1 for index in self.names if not isinstance(index, DotVariableIndex))

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

    @cached_method
    def degrees_by_position(self, grading_group: QuiverGradingGroup):
        result = [list() for _ in range(self.ngens())]
        for index, var in self.variables.items():
            if var.position is not None:
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

        y_transposed = dok_array((1, grading_group.dimension()), dtype=intc)
        for i, v in degree._vector_(sparse=True).dict(copy=False).items():
            y_transposed[0, i] = v

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
                    m.addConstr(A_non_inv @ x_non_inv == y_transposed)
                elif A_non_inv.shape[1] == 0:
                    m.addConstr(A_inv @ x_inv == y_transposed)
                else:
                    m.addConstr(A_inv @ x_inv + A_non_inv @ x_non_inv == y_transposed)

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

    def exp_degree(self, exp: ETuple, grading_group: QuiverGradingGroup):
        degrees_by_position = self.degrees_by_position(grading_group)
        return grading_group.linear_combination(
            (degrees_by_position[position], power)
            for position, power in exp.sparse_iter()
        )

    # libsingular wrapper in Sage does not allow to change elements for a wrapper,
    # so we define degree in the parent
    def element_degree(
        self, element, grading_group: QuiverGradingGroup, check_if_homogeneous=False
    ):
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
        return all(
            not deg.is_zero()
            for deg in self.degrees_by_position(grading_group)
        )

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
        return et[: self.number_of_non_dot_params]

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
            if self.quiver_data == other.quiver_data:
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

    @cached_method
    def hom_in_simple(self):
        """
        Kill dots, set all multiplicative parameters to 1.

        Gets scalars in a ring R.
        """
        variables_images = []
        for index, var in self.variables.items():
            if var.name is not None:
                if isinstance(index, DotVariableIndex):
                    # setting all dots to zero
                    variables_images.append(self.base().zero())
                else:
                    # setting all other parameters to 1
                    variables_images.append(self.base().one())

        return self.hom(variables_images, self.base())

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


class KLRWUpstairsDotsAlgebra(KLRWDotsAlgebra, PolynomialRing_sing):
    def init_algebra(self, base_ring, nondot_names_list, dot_names_list, order):
        super(KLRWDotsAlgebra, self).__init__(
            base_ring,
            len(nondot_names_list) + len(dot_names_list),
            nondot_names_list + dot_names_list,
            order,
        )


# class DotAlgebraElement_invertible_parameters(LaurentPolynomial_mpair):
    # TODO: can speed up __init__ by using reduce=False
    # need Cython
#    pass
    # def __getattribute__(self, name):
    #     print("Element :: Accessing attribute {}".format(name))
    #     return super().__getattribute__(name)


class KLRWUpstairsDotsAlgebra_invertible_parameters(
    KLRWDotsAlgebra, LaurentPolynomialRing
):
    # Element = DotAlgebraElement_invertible_parameters

    def init_algebra(self, base_ring, nondot_names_list, dot_names_list, order):
        polynomial_ring = PolynomialRing(
            base_ring,
            len(nondot_names_list) + len(dot_names_list),
            nondot_names_list + dot_names_list,
            order=order,
        )
        super(KLRWDotsAlgebra, self).__init__(polynomial_ring)

    # def __getattribute__(self, name):
    #    print("Algebra :: Accessing attribute {}".format(name))
    #    return super().__getattribute__(name)
