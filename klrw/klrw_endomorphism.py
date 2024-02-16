from sage.algebras.free_algebra import FreeAlgebra_generic
from sage.algebras.free_algebra_element import FreeAlgebraElement
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.categories.action import Action


class LeftKLRWEndomorphismAction(Action):
    def _act_(
        self, p: FreeAlgebraElement, x: IndexedFreeModuleElement
    ) -> IndexedFreeModuleElement:
        return self.codomain().linear_combination(
            (self._act_by_words_(word, x), coeff) for word, coeff in p
        )

    def _act_by_words_(self, word, x):
        sequence = list(word)
        result = self.domain().zero()
        for braid, poly in x:
            term = self.domain().term(braid, poly)
            state = braid.left_state()
            for index, n in reversed(sequence):
                for _ in range(n):
                    new_braid = self.actor().simple_endomorphisms_as_klrw_braid(
                        index, state, act_on_left=True
                    )
                    term = self.domain().monomial(new_braid) * term
                    state = state.act_by_s(new_braid[0])
            result += term
        return result


class RightKLRWEndomorphismAction(Action):
    def _act_(
        self, p: FreeAlgebraElement, x: IndexedFreeModuleElement
    ) -> IndexedFreeModuleElement:
        result = self.domain().zero()
        for word, coeff in p:
            for braid, poly in x * coeff:
                term = self.domain().term(braid, poly)
                state = braid.right_state()
                for index, n in word:
                    for _ in range(n):
                        new_braid = self.actor().simple_endomorphisms_as_klrw_braid(
                            index, state, act_on_left=False
                        )
                        term = term * self.domain().monomial(new_braid)
                        state = state.act_by_s(new_braid[0])
                result += term
        return result


# So far covering only A1 case; u and h are elements of the free algebra.
class KLRWEndomorphismAlgebra(FreeAlgebra_generic):
    def __init__(
        self,
        krlw_algebra,
        number_of_moving_strands,
        crossing_prefix="y",
        left_prefix="l",
        right_prefix="r",
    ):
        self.krlw_algebra = krlw_algebra
        self.crossing_prefix = crossing_prefix
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.number_of_moving_strands = number_of_moving_strands

        names_crossing = [
            crossing_prefix + repr(i) for i in range(1, number_of_moving_strands)
        ]
        names_left = [
            left_prefix + repr(i) for i in range(1, number_of_moving_strands + 1)
        ]
        names_right = [
            right_prefix + repr(i) for i in range(1, number_of_moving_strands + 1)
        ]

        super().__init__(
            self.krlw_algebra.base(),
            n=3 * number_of_moving_strands - 1,
            names=names_crossing + names_left + names_right,
        )

        self.simple_endomorphisms = {
            self(name).leading_item()[0]: (self.crossing_prefix, index + 1)
            for index, name in enumerate(names_crossing)
        }
        self.simple_endomorphisms |= {
            self(name).leading_item()[0]: (self.left_prefix, index + 1)
            for index, name in enumerate(names_left)
        }
        self.simple_endomorphisms |= {
            self(name).leading_item()[0]: (self.right_prefix, index + 1)
            for index, name in enumerate(names_right)
        }

    def simple_endomorphisms_as_klrw_braid(self, name, state, act_on_left):
        """
        If act_on_left is True, state will be the right state
        If act_on_left is False, state will be the left state
        """
        prefix, index = self.simple_endomorphisms[name]
        # find the moving strand of correct index
        current_index = 0
        for i in range(len(state)):
            if not state[i].is_framing():
                current_index += 1
            if current_index == index:
                break
        else:
            raise ValueError("Moving strand of index {} not found".format(index))

        if prefix == self.crossing_prefix:
            if state[i + 1].is_framing():
                raise ValueError("Impossible crossing")
            return self.krlw_algebra.KLRWBraid._element_constructor_(
                state, word=(i + 1,)
            )

        elif prefix == self.right_prefix and act_on_left:
            if i == 0:
                raise ValueError("Impossible crossing")
            if not state[i - 1].is_framing():
                raise ValueError("Impossible crossing")
            return self.krlw_algebra.KLRWBraid._element_constructor_(state, word=(i,))

        elif prefix == self.right_prefix and not act_on_left:
            if i + 1 == len(state):
                raise ValueError("Impossible crossing")
            if not state[i + 1].is_framing():
                raise ValueError("Impossible crossing")
            new_state = state.act_by_s(i + 1)
            return self.krlw_algebra.KLRWBraid._element_constructor_(
                new_state, word=(i + 1,)
            )

        elif prefix == self.left_prefix and act_on_left:
            if i + 1 == len(state):
                raise ValueError("Impossible crossing")
            if not state[i + 1].is_framing():
                raise ValueError("Impossible crossing")
            return self.krlw_algebra.KLRWBraid._element_constructor_(
                state, word=(i + 1,)
            )

        elif prefix == self.left_prefix and not act_on_left:
            if i == 0:
                raise ValueError("Impossible crossing")
            if not state[i - 1].is_framing():
                raise ValueError("Impossible crossing")
            new_state = state.act_by_s(i)
            return self.krlw_algebra.KLRWBraid._element_constructor_(
                new_state, word=(i,)
            )

    def _repr_(self):
        """
        How to print algebra self.
        """
        return "Endomorphism algebra of a KLRW algebra with {} moving strands.".format(
            self.number_of_moving_strands
        )
