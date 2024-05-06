from typing import NamedTuple
from collections import defaultdict
from dataclasses import dataclass, InitVar
from types import MappingProxyType

from sage.combinat.root_system.dynkin_diagram import DynkinDiagram_class
from sage.combinat.root_system.cartan_type import CartanType_abstract
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.polynomial.multi_polynomial_libsingular import (
    MPolynomialRing_libsingular as PolynomialRing,
)
from sage.rings.polynomial.polydict import ETuple
from sage.rings.ring import CommutativeRing, Ring
from sage.rings.integer_ring import ZZ
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.structure.sage_object import SageObject

from sage.misc.cachefunc import cached_method
from sage.misc.lazy_attribute import lazy_attribute
from sage.misc.misc_c import prod

from itertools import product


class NodeInFramedQuiver(NamedTuple):
    node: object
    framing: bool = False

    def is_framing(self):
        return self.framing

    def __repr__(self):
        if self.framing:
            return "W" + self.node.__repr__()
        else:
            return "V" + self.node.__repr__()

    def make_framing(self):
        return self._replace(framing=True)


class NonOrientedEdge(frozenset):
    def __new__(cls, vertex1, vertex2):
        assert isinstance(vertex1, NodeInFramedQuiver)
        assert isinstance(vertex2, NodeInFramedQuiver)
        return super(NonOrientedEdge, cls).__new__(cls, frozenset((vertex1, vertex2)))

    def vertices(self):
        if len(self) == 2:
            vertex1, vertex2 = self
        if len(self) == 1:
            (vertex1,) = self
            vertex2 = vertex1
        return vertex1, vertex2

    def __repr__(self):
        vertex1, vertex2 = self.vertices()
        return repr(vertex1) + "--" + repr(vertex2)

    def __reduce__(self):
        vertex1, vertex2 = self.vertices()
        return (
            self.__class__,
            (vertex1, vertex2),
        )


class NodeInFramedQuiverWithMarks(NamedTuple):
    unmarked_node: NodeInFramedQuiver
    mark: object

    def unmark(self):
        return self.unmarked_node

    def __repr__(self):
        return self.unmarked_node.__repr__() + "_" + self.mark.__repr__()


class FramedDynkinDiagram_class(UniqueRepresentation, DynkinDiagram_class):
    def __init__(self, t=None, **options):
        assert isinstance(t, CartanType_abstract)
        # add a property framing = False to all labels
        relabel_dict = {v: NodeInFramedQuiver(v) for v in t.index_set()}
        t = t.relabel(relabel_dict)

        super().__init__(t.dynkin_diagram())
        # add framing vertices
        for v in t.index_set():
            w = v.make_framing()
            self.add_vertex(w)
            # in this convention marks are negative the off-diagonal coefficients
            self.add_edge(v, w, 1)
            self.add_edge(w, v, 1)

    def _repr_(self):
        ct = self.cartan_type()
        is_linear = ct.type() == "A" or ct.type() == "B" or ct.type() == "C"
        is_linear &= ct.is_finite()
        if is_linear:
            framed_node = "□"
            node = "◯"
            n = ct.rank()
            if n == 0:
                return ""
            result = (
                "".join("{!s:4}".format(v.make_framing()) for v in ct.index_set())
                + "\n"
            )
            result += "   ".join(framed_node for i in range(n)) + "\n"
            result += "   ".join("|" for i in range(n)) + "\n"
            if ct.type() == "A" or n == 1:
                result += "---".join(node for i in range(n)) + "\n"
            elif ct.type() == "B":
                result += "---".join(node for i in range(1, n)) + "=>=" + node + "\n"
            elif ct.type() == "C":
                result += "---".join(node for i in range(1, n)) + "=<=" + node + "\n"
            result += "".join("{!s:4}".format(v) for v in ct.index_set())
        else:
            result = "Framed Quiver"

        return result  # +"\n"+"Framed Quiver"

    @cached_method
    def scalar_product_of_simple_roots(self, i, j):
        ct = self.cartan_type()
        sym = ct.symmetrizer()
        if not i.is_framing():
            return self[i, j] * sym[i]
        else:
            return self[j, i] * sym[j]

    def is_simply_laced(self):
        return self.cartan_type().is_simply_laced()

    @cached_method
    def non_framing_nodes(self):
        return self.cartan_type().index_set()

    def inject_nodes(self, scope=None, verbose=True):
        """
        Defines globally nodes
        """
        vs = [v.__repr__() for v in self.vertices()]
        gs = [v for v in self.vertices()]
        if scope is None:
            scope = globals()
        if verbose:
            print("Defining %s" % (", ".join(vs)))
        for v, g in zip(vs, gs):
            scope[v] = g

    def KLRW_vertex_param_names(self, vertex_prefix="r"):
        vertex_names = {}
        # iterating over non-framing vertices
        for v in self.non_framing_nodes():
            vertex_names[v] = vertex_prefix + "_{}".format(v.node)

        return vertex_names

    def KLRW_edge_param_names(self, edge_prefix="t", framing_prefix="u"):
        edge_names = {}
        for n1, n2, _ in self.edges():
            # should we add second set of framing variables? Or just rescale?
            if n2.is_framing():
                assert n1.node == n2.node
                edge_names[n1, n2] = framing_prefix + "_{}".format(n1.node)
            else:
                if not n1.is_framing():
                    edge_names[n1, n2] = edge_prefix + "_{}_{}".format(n1.node, n2.node)

        return edge_names

    def KLRW_special_edge_param_names(self, special_edge_prefix="s"):
        # these are half of the squares of lenghts normalized
        # to have squared length 2 for short roots
        ct = self.cartan_type()
        half_lengths_squared = ct.symmetrizer()
        special_edge_names = {}
        for n1, n2, w in ct.dynkin_diagram().edges():
            print(n1, n2, w)
            if not n1.is_framing() and not n2.is_framing():
                if w > 1:
                    a = half_lengths_squared[n1]
                    b = half_lengths_squared[n2]
                    p = 1

                    while w - a * p >= 0:
                        q = (w - a * p) / b
                        special_edge_names[n1, n2] = (p, q)
                        print(a, "*", p, " + ", b, "*", q, " = ", w)
                        p += 1

        return special_edge_names


@dataclass(repr=False)
class FramedDynkinDiagram_with_dimensions(SageObject):
    ct: InitVar[CartanType_abstract | None] = None
    quiver: FramedDynkinDiagram_class | None = None
    dimensions_dict: dict | defaultdict | MappingProxyType | None = None

    def __post_init__(self, ct: CartanType_abstract | None):
        if ct is not None:
            assert isinstance(ct, CartanType_abstract), (
                "Don't know how to make a Framed Dynkin Diagram"
                + " with dimensions from {}".format(ct.__class__)
            )
            if self.quiver is not None:
                assert self.quiver.cartan_type() == ct
            else:
                self.quiver = FramedDynkinDiagram_class(ct)
        else:
            assert self.quiver is not None, "Quiver must be given"

        if self.dimensions_dict is None:
            self.dimensions_dict = defaultdict(int)
        if isinstance(self.dimensions_dict, list):
            self.dimensions_dict = {key: value for key, value in self.dimensions_dict}

    def __hash__(self):
        """
        We need hash to be able pass instances of this class to __init__'s
        of UniqueRepresentation classes [e.g. KLRW algebra]
        """
        return hash((self.dimensions_list(), self.quiver.cartan_type()))

    #    def __getattr__(self, name):
    #        return getattr(self.quiver, name)

    def __getitem__(self, key):
        """
        If key is a NodeInFramedQuiver returns dimensions in the framing
        Otherwise returns an element of the Cartan matrix.
        """
        if isinstance(key, NodeInFramedQuiver):
            return self.dimensions_dict[key]
        else:
            return self.quiver[key]

    def __setitem__(self, key, dim):
        """
        A more convenient way to set dimentions
        """
        if isinstance(key, NodeInFramedQuiver):
            self.dimensions_dict[key] = dim
        if isinstance(key, tuple):
            node = NodeInFramedQuiver(*key)
        else:
            node = NodeInFramedQuiver(key)
        assert node in self.quiver.vertices()
        self.dimensions_dict[node] = dim

    def dimensions_list(self):
        return tuple(sorted(self.dimensions_dict.items()))

    def dimensions(self, copy=False):
        if copy:
            return self.dimensions_dict.copy()
        else:
            return self.dimensions_dict

    def get_dim(self, *index):
        """
        Gets the dimension of a node
        Allows to access
        """
        if len(index) == 1:
            if isinstance(index[0], NodeInFramedQuiver):
                return self.dimensions_dict[index]
            node = NodeInFramedQuiver(index[0], False)
            if node in self.quiver.cartan_type().index_set():
                return self.dimensions_dict[node]
        if len(index) == 2:
            assert isinstance(index[1], bool)
            return self.dimensions_dict[NodeInFramedQuiver(*index)]

        raise ValueError("Incorrect index")

    def set_dim(self, dim, index, is_framing=False):
        node = NodeInFramedQuiver(index, is_framing)
        assert node in self.quiver.vertices()
        self.dimensions_dict[node] = dim

    def scalar_product_of_simple_roots(self, *args, **kwargs):
        return self.quiver.scalar_product_of_simple_roots(*args, **kwargs)

    def is_simply_laced(self, *args, **kwargs):
        return self.quiver.is_simply_laced(*args, **kwargs)

    def non_framing_nodes(self, *args, **kwargs):
        return self.quiver.non_framing_nodes(*args, **kwargs)

    def inject_nodes(self, *args, **kwargs):
        self.quiver.inject_nodes(*args, **kwargs)

    def KLRW_vertex_param_names(self, *args, **kwargs):
        return self.quiver.KLRW_vertex_param_names(*args, **kwargs)

    def KLRW_edge_param_names(self, *args, **kwargs):
        return self.quiver.KLRW_edge_param_names(*args, **kwargs)

    def KLRW_special_edge_param_names(self, *args, **kwargs):
        return self.quiver(*args, **kwargs)

    def _repr_(self):
        ct = self.quiver.cartan_type()
        is_linear = ct.type() == "A" or ct.type() == "B" or ct.type() == "C"
        is_linear &= ct.is_finite()
        if is_linear:
            result = (
                "".join(
                    "{!s:4}".format(self.dimensions_dict[v])
                    for v in self.quiver.vertices()
                    if v.is_framing()
                )
                + "\n"
            )
            result += self.quiver._repr_() + "\n"
            result += "".join(
                "{!s:4}".format(self.dimensions_dict[v])
                for v in self.quiver.vertices()
                if not v.is_framing()
            )
        else:
            result = "Framed Quiver of type {}{}".format(ct.type(), ct.rank())
        return result

    def KLRW_dots_names(self, dots_prefix="x"):
        dots_names = {}
        # iterating over non-framing vertices
        for v in self.quiver.cartan_type().index_set():
            for k in range(self.dimensions_dict[v]):
                dots_names[v, k + 1] = dots_prefix + "_{}_{}".format(v.node, k + 1)
        return dots_names

    def KLRW_deformations_names(self, deformations_prefix="z"):
        deformations_names = {}
        for v in self.quiver.cartan_type().index_set():
            w = v.make_framing()
            for k in range(self.dimensions_dict[w]):
                deformations_names[w, k + 1] = deformations_prefix + "_{}_{}".format(
                    w.node, k + 1
                )
        return deformations_names

    def names(
        self,
        no_deformations=True,
        no_vertex_parameters=False,
        no_edge_parameters=False,
        **prefixes,
    ):
        """
        Returns a dictionary {label: name}
        """
        # We need the dictionary to keep the order of elements
        # In Python 3.7 and higher it is guaranteed
        # We also use |= from Python 3.9 and higher
        names = {}
        if not no_vertex_parameters:
            names |= self.KLRW_vertex_param_names(prefixes.get("vertex_prefix", "r"))
        if not no_edge_parameters:
            names |= self.KLRW_edge_param_names(
                prefixes.get("edge_prefix", "t"), prefixes.get("framing_prefix", "u")
            )
        names |= self.KLRW_dots_names(prefixes.get("dots_prefix", "x"))
        if not no_deformations:
            names |= self.KLRW_deformations_names(
                prefixes.get("deformations_prefix", "z")
            )

        return names

    def immutable_copy(self):
        return FramedDynkinDiagram_with_dimensions_immutable(
            quiver=self.quiver,
            dimensions_dict=self.dimensions_dict,
        )


class FramedDynkinDiagram_with_dimensions_immutable(
    FramedDynkinDiagram_with_dimensions
):
    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)
        self.dimensions_dict = MappingProxyType(self.dimensions_dict)

    def __hash__(self):
        """
        We need hash to be able pass instances of this class to __init__'s
        of UniqueRepresentation classes [e.g. KLRW algebra]
        """
        return hash((self.dimensions_list(), self.quiver.cartan_type()))

    def __setitem__(self, key, dim):
        raise AttributeError("The dimensions are immutable")

    def set_dim(self, dim, index, is_framing=False):
        raise AttributeError("The dimensions are immutable")

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.quiver.cartan_type(),
                self.quiver,
                dict(self.dimensions_dict),
            ),
        )


class QuiverGradingGroupElement(IndexedFreeModuleElement):
    def ordinary_grading(self, as_scalar=True):
        parent = self.parent()
        equivariant_grading_name = parent.equivariant_grading_name
        if as_scalar:
            return self.coefficient(equivariant_grading_name)
        else:
            basis_vector = parent.monomial(equivariant_grading_name)
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
        **kwrds,
    ):
        self.equivariant_grading_name = equivariant_grading_name
        self.quiver = quiver
        names = []
        if vertex_scaling:
            self.vertices = quiver.non_framing_nodes()
            names += self.vertices
        if edge_scaling:
            # we keep only non-oriented pairs and remove doubles
            self.edges = set(NonOrientedEdge(v1, v2) for v1, v2, _ in quiver.edges())
            names += list(self.edges)
        names += [equivariant_grading_name]

        if "prefix" not in kwrds:
            kwrds["prefix"] = "d_"
        if "bracket" not in kwrds:
            kwrds["bracket"] = "{"

        super().__init__(R, names, element_class=QuiverGradingGroupElement, **kwrds)

    def crossing_grading(self, vertex1, vertex2):
        result = self.term(
            self.equivariant_grading_name,
            -self.quiver.scalar_product_of_simple_roots(vertex1, vertex2),
        )
        if vertex1 == vertex2:
            if vertex1 in self.indices():
                result += self.monomial(vertex1)
        elif NonOrientedEdge(vertex1, vertex2) in self.indices():
            result += self.monomial(NonOrientedEdge(vertex1, vertex2))
        return result

    def dot_algebra_grading(self, index):
        if is_a_dot_index(index, only_non_framing=False):
            if not index[0].is_framing():
                return self.term(
                    self.equivariant_grading_name,
                    self.quiver.scalar_product_of_simple_roots(index[0], index[0]),
                )
        elif isinstance(index, NodeInFramedQuiver):
            if index in self.indices():
                return self.monomial(index)
        else:
            assert len(index) == 2
            edge = NonOrientedEdge(*index)
            if edge in self.indices():
                return 2 * self.monomial(edge)

        return self.zero()

    '''
    @staticmethod
    def coerce_map(parent, x):
        mon_coeff = {}
        for index, coeff in x._monomial_coefficients:
            if isinstance(index, NonOrientedEdge):
                if index in self.edges:
                    mon_coeff[index] = coeff
                else:
                    raise ValueError(
                        "No edge {} in quiver for grading".format(index)
                    )
            if isinstance(index, NodeInFramedQuiver):
                if index in self.vertices:
                    mon_coeff[index] = coeff
                else:
                    raise ValueError(
                        "No vertex {} in quiver for grading".format(index)
                    )
        return

    def _coerce_map_from_(self, G):
        """
        Return coerse map from gradings for similar quivers
        """
        if isinstance(G, QuiverGradingGroup):
            # assert that quivers match?

            try:
                CR = R.base_extend(self.base_ring())
            except (NotImplementedError, TypeError):
                pass
            else:
                if CR == self:
                    return lambda parent, x: self._from_dict(
                        x._monomial_coefficients, coerce=True, remove_zeros=True
                    )
        else:
            raise ValueError("Don't know thow to coerce from {}".format(G.__class__))
    '''


@dataclass(slots=True)
class QuiverParameter:
    # index : object
    name: str
    position: int
    monomial: object

    def __repr__(self):
        return self.monomial.__repr__()


def is_a_dot_index(index, only_non_framing=True):
    if not isinstance(index, NodeInFramedQuiver):
        if not isinstance(index[1], NodeInFramedQuiver):
            if not only_non_framing:
                return True
            if not index[0].is_framing():
                return True
    return False


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
    # have to define self.variables
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
                if (n1, n2) not in self.variables:
                    self.variables[n1, n2] = QuiverParameter(None, None, self.one())
        else:
            for n1, n2 in product(quiver.vertices(), quiver.vertices()):
                self.variables[n1, n2] = QuiverParameter(
                    None, None, self(default_edge_parameter)
                )
        if default_vertex_parameter is not None:
            for key in quiver.cartan_type().index_set():
                self.variables[key] = QuiverParameter(
                    None, None, self(default_vertex_parameter)
                )
        if no_deformations:
            for key in quiver_data.KLRW_deformations_names():
                self.variables[key] = QuiverParameter(None, None, self.zero())

    def center_gens(self):
        for index, variable in self.variables.items():
            if not isinstance(index, NodeInFramedQuiver):
                if not isinstance(index[1], NodeInFramedQuiver):
                    if not index[0].is_framing():
                        continue
            if variable.name is not None:
                yield variable.monomial
        yield from self.symmetric_dots_gens()

    def symmetric_dots_gens(self):
        raise NotImplementedError()

    def basis_modulo_symmetric_dots(self):
        variables = []
        degree_ranges = []
        for index, variable in self.variables.items():
            if is_a_dot_index(index):
                degree_ranges.append(range(index[1]))
                variables.append(variable.monomial)
        degree_iterator = product(*degree_ranges)
        for degree in degree_iterator:
            yield prod(var**deg for var, deg in zip(variables, degree))


class KLRWUpstairsDotsAlgebra(PolynomialRing, KLRWDotsAlgebra):
    def __init__(
        self,
        base_ring,
        quiver_data: FramedDynkinDiagram_with_dimensions,
        no_deformations=True,
        default_vertex_parameter=None,
        default_edge_parameter=None,
        order="degrevlex",
        **prefixes,
    ):
        # remember initialization data for pickle/unpickle
        self.quiver_data = quiver_data
        self.quiver = quiver_data.quiver
        self.no_deformations = no_deformations
        self.default_vertex_parameter = default_vertex_parameter
        self.default_edge_parameter = default_edge_parameter
        self.prefixes = prefixes

        no_vertex_parameters = default_vertex_parameter is not None
        no_edge_parameters = default_edge_parameter is not None
        self.names = quiver_data.names(
            no_deformations=no_deformations,
            no_vertex_parameters=no_vertex_parameters,
            no_edge_parameters=no_edge_parameters,
            **prefixes,
        )

        names_list = [name for _, name in self.names.items()]
        super().__init__(base_ring, len(names_list), names_list, order)

        # type of variables is uniquely determined by its index data
        self.variables = {
            index: QuiverParameter(name, position, self(name))
            for position, (index, name) in enumerate(self.names.items())
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
                self.base_ring(),
                self.quiver_data,
                self.no_deformations,
                self.default_vertex_parameter,
                self.default_edge_parameter,
                self.term_order().name(),
            ),
        )

    @lazy_attribute
    def number_of_non_dot_params(self):
        return sum(1 for index in self.names if not is_a_dot_index(index))

    @cached_method
    def dot_exptuples_and_degrees(self, grading_group: QuiverGradingGroup):
        result = {}
        relevant_grading_index = grading_group.equivariant_grading_name
        for index, var in self.variables.items():
            if is_a_dot_index(index):
                exps = var.monomial.exponents(as_ETuples=True)
                # must be a monomial
                assert len(exps) == 1
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
            if is_a_dot_index(index):
                result[index[0]].append(variable)

        # make everything immutable
        for key in result:
            result[key] = tuple(result[key])

        return MappingProxyType(result)

    def symmetric_dots_gens(self):
        elementary_symmetric_polynomials = {}
        for index, variable in self.variables.items():
            if is_a_dot_index(index):
                if index[0] in elementary_symmetric_polynomials:
                    e_list = elementary_symmetric_polynomials[index[0]]
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
                    elementary_symmetric_polynomials[index[0]] = [variable.monomial]

        for _, e_list in elementary_symmetric_polynomials.items():
            yield from e_list

    def basis_by_degree(self, degree):
        return tuple(self.monomial(*et) for et in self.exps_by_degree(degree))

    @cached_method
    def exps_by_degree(self, degree):
        result = self.one()
        for index, coeff in degree:
            if isinstance(index, NodeInFramedQuiver):
                if coeff < 0:
                    return tuple()
                result *= self.variables[index].monomial ** coeff
            elif isinstance(index, NonOrientedEdge):
                if int(coeff) % int(2) != 0 or coeff < 0:
                    return tuple()
                d = int(coeff) // int(2)
                v1, v2 = index
                if v2.is_framing():
                    new_part = self.variables[v1, v2].monomial ** d
                elif v1.is_framing():
                    new_part = self.variables[v2, v1].monomial ** d
                else:
                    new_part = sum(
                        self.variables[v1, v2].monomial ** k
                        * self.variables[v2, v1].monomial ** (d - k)
                        for k in range(d + 1)
                    )
                result *= new_part
            elif index == degree.parent().equivariant_grading_name:
                if int(coeff) % int(2) != 0 or coeff < 0:
                    return tuple()
                new_part = sum(
                    self.monomial(*etuple)
                    for etuple in exps_for_dots_of_degree(
                        self.dot_exptuples_and_degrees(degree.parent()),
                        coeff,
                    )
                )
                result *= new_part
            else:
                raise ValueError("Unknown type of grading: {}".format(index))
        return tuple(x for x in result.exponents())

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
                    raise ValueError("The dots are not homogeneous!")

        return degree

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

    @cached_method
    def hom_in_simple(self):
        """
        Kill dots, set all multiplicative parameters to 1.

        Gets scalars in a ring R.
        """
        variables_images = []
        for index, var in self.variables.items():
            if var.name is not None:
                if is_a_dot_index(index):
                    # setting all dots to zero
                    variables_images.append(self.base().zero())
                else:
                    # setting all other parameters to 1
                    variables_images.append(self.base().one())

        return self.hom(variables_images, self.base())
