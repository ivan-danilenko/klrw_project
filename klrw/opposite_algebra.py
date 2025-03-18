from sage.structure.element import ModuleElement, Element
from sage.modules.module import Module
from sage.structure.unique_representation import UniqueRepresentation
from sage.misc.lazy_attribute import lazy_attribute


class OppositeAlgebraElement(ModuleElement):
    """
    The class for the elements of the opposite algebra.

    The multiplication is changed to the opposite one.
    Left action by scalars is changed to the right one
    and vice versa.
    Implemented as a wrapper.
    """

    __slots__ = ("value",)

    def __init__(self, parent, value):
        ModuleElement.__init__(self, parent=parent)
        self.value = value

    def _add_(self, other):
        return self.__class__(
            self.parent(),
            self.value + other.value,
        )

    def _rmul_(self, left: Element):
        """
        The right action by
        the original algebra elements
        is the original right action,
        not swapped.
        Otherwise it's not a right action.
        """
        return self.__class__(
            self.parent(),
            left * self.value,
        )

    def _lmul_(self, right: Element):
        """
        The left action by
        the original algebra elements
        is the original left action,
        not swapped.
        Otherwise it's not a left action.
        """
        return self.__class__(
            self.parent(),
            self.value * right,
        )

    def _mul_(left, right):
        """
        We define the opposite multiplication.
        """
        return left.__class__(
            left.parent(),
            right.value * left.value,
        )

    def _repr_(self):
        return repr(self.value)

    def is_zero(self):
        return self.value.is_zero()

    def _richcmp_(left, right, op: int):
        return left.value._richcmp_(right.value, op)

    def __hash__(self):
        return hash(self.value)

    def __reduce__(self):
        """
        For pickling.
        """
        return (
            self.__class__, (self.parent(), self.value)
        )


class OppositeAlgebra(UniqueRepresentation, Module):
    """
    Endomorphism algebra of a free rank one module.

    The algebra is assumed to be unital, but not
    necessary commutative.
    Then every endomorphism of a rank one free module
    is given by *right* multiplication:
    `x |-> x*a`
    for some `a` in algebra. Explicitly, this `a`
    is the image of the identity.
    Note that then `End` becomes the algebra opposite
    to the original algebra because
    the maps compose in the opposite way:
    `f_a: x |-> x*a`
    and
    `f_b: x |-> x*b`
    have composition `f_b * f_a` given by `a*b`:
    `x |-> x*a |-> (x*a)*b = x*(a*b)`.

    So, we define the opposite algebra and act by
    scalars from the opposite sides.

    We implement isomorphism [of modules over the center]
    from and to the algebra.
    Since it is not save to allow coercions
    [multiplication is not respected by these isomorphisms]
    we only allow conversions, i.e. explicit transformations.

    Elements can be constructed by passing to __call__ or
    _element_constructor_ the same data as to same methods
    of the algebra
    """

    Element = OppositeAlgebraElement

    def __init__(self, algebra):
        self.algebra = algebra
        base = algebra.base()

        from sage.categories.algebras import Algebras

        category = Algebras(base) & algebra.category()

        Module.__init__(self, base=base, category=category)

    def _element_constructor_(self, *args, **kwargs):
        x = self.algebra(*args, **kwargs)
        return self.isomorphism_from_algebra(x)

    def _coerce_map_from_(self, other):
        if isinstance(other, OppositeAlgebra):
            if self.algebra.has_coerce_map_from(other.algebra):
                from sage.categories.morphism import SetMorphism
                from sage.categories.homset import Hom

                morphism = SetMorphism(
                    Hom(other, self),
                    lambda x, self=self: self.element_class(
                        self, self.algebra(x.value)
                    ),
                )
                return morphism
        return None

    def _convert_map_from_(self, algebra):
        if self.algebra.has_coerce_map_from(algebra):
            from sage.categories.morphism import SetMorphism
            from sage.categories.homset import Hom

            morphism = SetMorphism(
                Hom(self.algebra, self),
                lambda x, self=self: self.element_class(self, self.algebra(x)),
            )
            return morphism
        if isinstance(algebra, OppositeAlgebra):
            from sage.categories.morphism import SetMorphism
            from sage.categories.homset import Hom

            morphism = SetMorphism(
                Hom(algebra, self),
                lambda x, self=self: self.element_class(
                    self, self.algebra(x.value)
                ),
            )
            return morphism
        return None

    @lazy_attribute
    def isomorphism_to_algebra(self):
        """
        Isomorphism to the algebra.

        `(x |-> x*a) |-> a`
        This is an isomorphism of modules over the center.
        """
        from sage.categories.morphism import SetMorphism
        from sage.categories.homset import Hom

        isomorphism = SetMorphism(
            Hom(self, self.algebra),
            lambda x: x.value,
        )

        return isomorphism

    @lazy_attribute
    def isomorphism_from_algebra(self):
        """
        Isomorphism from the algebra.

        `a |-> (x |-> x*a)`
        This is an isomorphism of modules over the center.
        """
        return self._convert_map_from_(self.algebra)

    def one(self):
        return self(self.algebra.one())

    def zero(self):
        return self(self.algebra.zero())

    def from_base_ring(self, r):
        """
        Return the canonical embedding of ``r`` into ``self``.
        """
        return self(self.algebra.from_base_ring(r))

    def _repr_(self):
        result = "Endomorphism algebra of a free rank one module over "
        result += repr(self.algebra)

        return result
