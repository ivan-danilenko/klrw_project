from sage.combinat.free_module import CombinatorialFreeModule

# for an example
from sage.categories.action import Action
from sage.rings.complex_mpfr import ComplexField
from sage.categories.finite_dimensional_algebras_with_basis import (
    FiniteDimensionalAlgebrasWithBasis,
)
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.misc.lazy_attribute import lazy_attribute
import operator


class LeftFreeBimoduleMonoid(CombinatorialFreeModule):
    """
    This is a generalizations of an algebra over a field [represented in a fixed basis]
    One way to describe it is a monoidal object in the category of R-bimodules for some
    ring R that is free as a left R-module.
    If R is a (commutative) field, this is an algebra.
    Less formally, it's a ring A that is also an R-bimodule for a given ring R.
    Compatibility of the structures comes from multiplication being left linear in
    the first argument, right bilinear in the second argument:
    (ra)*b = r(a*b)
    a*(br) = (a*b)r
    [here r is in R, a,b are in A]
    Also right action on the first argument can be transformed into a left action on
    the second argument:
    (ar)*b = a*(rb)
    -----
    This happens whenever R is a subring of A [with the assumption that A is free
    as a left R-module]
    The classical notion of algebra refers to the case when R is *in the center* of A
    [and, usually, R is a field].
    A classical example is quaternions. Complex numbers are a subring, but quaternions
    is not an algebra over complex numbers.
    """

    def _product_from_product_on_basis_multiply(self, left, right):
        r"""
        Compute the product of two elements by extending
        bilinearly the method :meth:`product_on_basis`
        and using the right action by the base ring.
        [can define in :meth:`_get_action_`]
        When we compute (sa)*(rb) with r,s in the base ring,
        we write s((ar)*b).
        """
        return self.linear_combination(
            self._product_from_product_on_basis_iter_(left, right)
        )

    def _product_from_product_on_basis_iter_(self, left, right):
        r"""
        Iterator for `_product_from_product_on_basis_multiply`.

        Iterates over `(index, coefficient)` in the product
        of two elements (possibly with repretitions).
        When we compute (sa)*(rb) with r,s in the base ring,
        we write s((ar)*b).
        The product (ar) is given by the iterator `mid_iterator`
        running through pairs `(mon_mid, coeff_mid)`
        """
        for (mon_left, coeff_left) in left:
            for (mon_right, coeff_right) in right:
                mid_iterator = self.right_base_action._act_on_base_in_bimodule_iter_(
                    coeff_right,
                    mon_left,
                )
                for (mon_mid, coeff_mid) in mid_iterator:
                    yield (
                        self.product_on_basis(mon_mid, mon_right),
                        coeff_left * coeff_mid
                    )

    @lazy_attribute
    def right_base_action(self):
        return self._right_base_action_(self)

    def _right_base_action_(self, other):
        """
        To be implemented in subclasses.

        Returns an instance of a class subclassing `RightActionOnBimodule`
        if `other` can be coerced to `self.base()`
        """
        raise NotImplementedError()

    def _get_action_(self, other, op, self_on_left):
        # the new action needed only if we act from the right
        if op == operator.mul and self_on_left:
            if self.base().has_coerce_map_from(other):
                return self._right_base_action_(other)

        return super()._get_action_(other, op, self_on_left)

    # TODO: redefine linear_combination to get properly working "scalars on left" option


class RightActionOnBimodule(Action):
    def _act_(self, p, x: IndexedFreeModuleElement) -> IndexedFreeModuleElement:
        return self.codomain().sum_of_terms(
            self._act_iter_(p, x)
        )

    def _act_iter_(self, p, x: IndexedFreeModuleElement):
        """
        Same as `_act_`, but iterates over pairs `(index, coefficient)`
        instead of making an element out of them.
        """
        for index, left_coeff in x:
            for new_index, coeff in self._act_on_base_in_bimodule_iter_(p, index):
                yield (new_index, left_coeff * coeff)

    def _act_on_base_in_bimodule_iter_(self, p, x_index):
        """
        To be implemented in subclasses.

        This is the generator of pairs `(index, coefficient)`
        if the right multiplication of a basis vector with index `x_index`
        and base ring element `p`.
        """
        raise NotImplementedError()


"""
An example: quaternions
"""


class CAction(RightActionOnBimodule):
    def _act_on_base_in_bimodule_iter_(self, g, x_index):
        g = self.codomain().base()(g)
        if x_index == "e":
            yield (x_index, g)
        elif x_index == "j":
            yield (x_index, g.conjugate())
        else:
            raise ValueError("Unknown basis vector {}", x_index)


class Quaternion(LeftFreeBimoduleMonoid):
    def __init__(self):
        CC = ComplexField()
        super().__init__(
            CC, ["e", "j"], category=FiniteDimensionalAlgebrasWithBasis(CC)
        )

    def _right_base_action_(self, other):
        return CAction(other, self, is_left=False, op=operator.mul)

    def product_on_basis(self, left, right):
        if left == "e":
            if right == "j":
                return self.monomial("j")
            elif right == "e":
                return self.monomial("e")
            else:
                raise ValueError()
        elif left == "j":
            if right == "j":
                return -self.monomial("e")
            elif right == "e":
                return self.monomial("j")
            else:
                raise ValueError()
