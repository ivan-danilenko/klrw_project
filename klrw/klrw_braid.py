from collections.abc import Iterable
from typing import NamedTuple
from itertools import chain, count

from sage.structure.element_wrapper import ElementWrapper
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.parent import Set_generic
from sage.misc.cachefunc import cached_method

from .framed_dynkin import FramedDynkinDiagram_with_dimensions
from .klrw_state import KLRWstate_set, KLRWstate


class KLRWbraid_data(NamedTuple):
    """
    Class representing a braid in KLRW algebra.

    state_ is a tuple of braid colors
    word_ is a tuple of indices of crossings starting with the state_
    Default is word_=()

    Conventions:
    Word starts from the state_
    state_ is on the left
    Here we call "left" the side that is on the left under product
    """

    state_: KLRWstate
    word_: tuple = ()


class KLRWbraid(ElementWrapper):
    wrapped_class = KLRWbraid_data
    __lt__ = ElementWrapper._lt_by_value
    # slots save memory by not allowing other
    __slots__ = ()

    def __init__(self, parent, state, word=()):
        value = self.wrapped_class(state, word)
        super().__init__(parent, value)

    def __getitem__(self, key):
        raise NotImplementedError()

    def _add_(left, right):
        raise NotImplementedError()

    def word(self):
        return self.value.word_

    def state(self):
        return self.value.state_

    def left_state(self):
        raise NotImplementedError()

    def right_state(self):
        raise NotImplementedError()

    def intersection_colors(self, i, closer_to_right=True):
        raise NotImplementedError()

    def find_intersection(self, i, j, right_state=True):
        raise NotImplementedError()

    def find_position_on_other_side(
        self, index: int, reverse: bool = False
    ):
        return self.parent().find_position_on_other_side(
            index=index, word=self.word(), reverse=reverse
        )

    def add_right_intersection(self, i):
        raise NotImplementedError()

    def ascii_art(self):
        n = len(self.state())
        result = "".join("{!s:4}".format(v) for v in self.right_state()) + "\n"
        result += "   ".join("|" for i in range(n)) + "\n"
        for ind, j in zip(count(start=1), reversed(self.word())):
            result += "|   " * (j - 1)
            result += " \\ / "
            result += "   |" * (n - j - 1) + "\n"
            result += "|   " * (j - 1)
            result += "  X  "
            result += "   |" * (n - j - 1) + "\n"
            result += "|   " * (j - 1)
            result += " / \\ "
            result += "   |" * (n - j - 1) + "\n"
            if ind != len(self.word()):
                if self.word()[-ind] == self.word()[-ind - 1] + 1:
                    result += "   ".join("|" for i in range(j - 1))
                    result += "   /   "
                    result += "   ".join("|" for i in range(n - j)) + "\n"
                elif self.word()[-ind] == self.word()[-ind - 1] - 1:
                    result += "   ".join("|" for i in range(j))
                    result += "   \\   "
                    result += "   ".join("|" for i in range(n - j - 1)) + "\n"
                else:
                    result += "   ".join("|" for i in range(n)) + "\n"

        result += "   ".join("|" for i in range(n)) + "\n"
        result += "".join("{!s:4}".format(v) for v in self.left_state()) + "\n"

        return result

    def print_ascii_art(self):
        print(self.ascii_art())


# TODO: does deriving from a tuple or making a C-class works better?
class KLRWbraid_state_on_left(KLRWbraid):
    def left_state(self):
        return self.value.state_

    def right_state(self):
        return self.parent().find_state_on_other_side(
            self.value.state_, self.value.word_
        )

    def intersection_colors(self, i, closer_to_right=True):
        intersection_position = self.word()[i]
        if i == 0:
            left_strand_position = intersection_position - 1
            right_strand_position = intersection_position
        else:
            word_to_the_state = self.word()[:i]
            left_strand_position = self.parent().find_position_on_other_side(
                intersection_position - 1, word_to_the_state, reverse=True
            )
            right_strand_position = self.parent().find_position_on_other_side(
                intersection_position, word_to_the_state, reverse=True
            )

        if closer_to_right:
            return (
                self.value.state_[right_strand_position],
                self.value.state_[left_strand_position],
            )
        else:
            return (
                self.value.state_[left_strand_position],
                self.value.state_[right_strand_position],
            )


class KLRWbraid_state_on_right(KLRWbraid):
    def left_state(self):
        return self.parent().find_state_on_other_side(
            self.value.state_, self.value.word_, reverse=True
        )

    def right_state(self):
        return self.value.state_

    def __getitem__(self, key):
        if isinstance(key, slice):
            # print(key)
            if key.step == 1 or key.step is None:
                if key.stop is None:
                    new_state = self.value.state_
                else:
                    leftover_word = self.value.word_[key.stop :]
                    new_state = self.parent().find_state_on_other_side(
                        self.value.state_, leftover_word, reverse=True
                    )
                return self.parent()._element_constructor_(
                    state=new_state, word=self.value.word_[key]
                )
            elif key.step == -1:
                raise ValueError(
                    "Inverted word may fail to be lex maximal,"
                    + " so we prohibit inverting words"
                )
            else:
                raise ValueError("Invalid step")
        else:
            return self.value.word_[key]

    def _add_(left, right):
        # TODO: add checks?
        return right.parent()._element_constructor_(
            state=right.right_state(), word=left.word() + right.word()
        )

    def add_right_intersection(self, i):
        return self.parent()._element_constructor_(
            state=self.value.state_.act_by_s(i), word=self.value.word_ + (i,)
        )

    def intersection_colors(self, i, closer_to_right=True):
        intersection_position = self.word()[i]
        if i == -1 or i == len(self.word()):
            left_strand_position = intersection_position - 1
            right_strand_position = intersection_position
        else:
            word_to_the_state = self.word()[i + 1 :]
            left_strand_position = self.parent().find_position_on_other_side(
                intersection_position - 1, word_to_the_state
            )
            right_strand_position = self.parent().find_position_on_other_side(
                intersection_position, word_to_the_state
            )

        if closer_to_right:
            return (
                self.value.state_[left_strand_position],
                self.value.state_[right_strand_position],
            )
        else:
            return (
                self.value.state_[right_strand_position],
                self.value.state_[left_strand_position],
            )

    def find_intersection(self, i, j, right_state=True):
        """
        Finds the index of the intersection of strands on i-th and j-th position
        i and j are indices in right state if right_state==True,
        in the left state otherwise.
        Returns -1 if there is no intersection
        """
        if right_state:
            for s, index in zip(
                reversed(self.word()), range(len(self.word()) - 1, -1, -1)
            ):
                if i == s - 1:
                    if i + 1 == j:
                        # found an intersection
                        return index
                    i += 1
                elif i == s:
                    i -= 1

                if j == s - 1:
                    j += 1
                elif j == s:
                    j -= 1
            return -1
        else:
            raise NotImplementedError()

    def intersections_with_given_strand(self, i):
        """
        Iterator returning (index, color, right_to_left)
        self.word[index] is the letter corresponding to the intersection
        color is the color of the other strand
        right_to_left is True if i-th strand moves right-to-left in this intersection
        [using the standard direction in the word]
        """
        for s, index in zip(reversed(self.word()), range(len(self.word()) - 1, -1, -1)):
            if i == s - 1:
                j = self.parent().find_position_on_other_side(
                    s, word=self.word()[index + 1 :], reverse=False
                )
                color = self.right_state()[j]
                yield index, color
                i += 1
            elif i == s:
                j = self.parent().find_position_on_other_side(
                    s - 1, word=self.word()[index + 1 :], reverse=False
                )
                color = self.right_state()[j]
                yield index, color
                i -= 1

    def position_for_new_s_from_right(self, i) -> tuple:
        """
        We have to check that the length of the product does not drop.
        I.e. that i-th and (i+1)-st strands have not crossed.

        Returns a tuple (index,position), so in the new word
        new_word[index] = position
        new_word[i] = self.word()[i] if i<index
        new_word[i+1] = self.word()[i] if i>index
        """
        # In the shortest form each strand is allowed to move from right to left
        # only in a single block of the word.
        # We need to find when right strand moves to the left and add
        # the new crossing to its run.
        # We track which strand is moving to the left at the moment.
        word = self.word()
        current_right_strand = self.parent().find_position_on_other_side(
            i, word=word, reverse=True
        )
        moving_strand = -1
        for j in range(len(word)):
            if word[j] == moving_strand - 1:
                moving_strand -= 1
            else:
                moving_strand = word[j]
                # if we see that the right strand never moves more to the right
                if moving_strand > current_right_strand:
                    # add one more crossing at the end of the run of the right strand
                    return j, current_right_strand
            if current_right_strand == word[j] - 1:
                current_right_strand += 1
            elif current_right_strand == word[j]:
                current_right_strand -= 1
        # if we end with the right strand still moving
        return len(word), i


class KLRWbraid_set(UniqueRepresentation, Set_generic):
    """
    ...
    """

    def _element_constructor_(self, state, word=()):
        if self._check:
            assert isinstance(state, KLRWstate)
            assert isinstance(word, tuple)
        return self.element_class(self, state, word)

    def __init__(
        self,
        framed_quiver: FramedDynkinDiagram_with_dimensions | None = None,
        state_on_right=True,
    ):
        """
        Dimentions have to be given as a list of instances of NodeWithDimensions
        """
        if state_on_right:
            self.Element = KLRWbraid_state_on_right
        else:
            self.Element = KLRWbraid_state_on_left
        self.KLRWstate_set = KLRWstate_set(framed_quiver)
        self._check = False

    def enable_checks(self):
        self.KLRWstate_set.enable_checks()
        self._check = True

    def disable_checks(self):
        self.KLRWstate_set.disable_checks()
        self._check = False

    @cached_method
    def find_state_on_other_side(
        self, state: KLRWstate, word: Iterable[int], reverse: bool = False
    ):
        if reverse:
            w = reversed(word)
        else:
            w = word
        # a fast way is the following
        new_state = state
        for i in w:
            new_state = new_state.act_by_s(i)
        return new_state

    # @cached_method
    def find_position_on_other_side(
        self, index: int, word: Iterable[int], reverse: bool = False
    ):
        """
        Tracks what how position of a single strand changes
        """
        if reverse:
            w = reversed(word)
        else:
            w = word
        for i in w:
            if index == i - 1:
                index += 1
            elif index == i:
                index -= 1
        return index

    def _braids_with_left_state_iter_(self, state):
        length = len(state)
        if length <= 1:
            yield self._element_constructor_(state=state, word=())
        else:
            last_strand_is_framing = state[-1].is_framing()
            last_strand_piece = state[-1:]
            # we don't want to check the dimensions in the recursion
            generic_braid_set = self.__class__()
            for truncated_braid in generic_braid_set._braids_with_left_state_iter_(
                state[:-1]
            ):
                new_state = self.KLRWstate_set._element_constructor_(
                    chain(truncated_braid.state(), last_strand_piece)
                )
                braid = self._element_constructor_(
                    state=new_state, word=truncated_braid.word()
                )
                yield braid
                for i in range(length - 1, 0, -1):
                    # two framings should not intersect
                    if last_strand_is_framing:
                        if braid.right_state()[i - 1].is_framing():
                            break
                    braid = braid.add_right_intersection(i)
                    yield braid

    def __iter__(self):
        for state in self.KLRWstate_set:
            yield from self._braids_with_left_state_iter_(state)

    def total_number_of_strands(self):
        return self.KLRWstate_set.total_number_of_strands()

    def braid_for_one_strand(
        self,
        right_state: KLRWstate,
        right_moving_strand_position: int,
        left_moving_strand_position: int,
        check=False,
    ) -> KLRWbraid:
        """
        Returns a KLRW braid with given right state
        where the only one strand moves from
        the position prescribed in
        left_moving_strand_position on the lest to
        right_moving_strand_position on the right.

        Works faster than the function for more general number of strands
        """
        if check:
            assert right_moving_strand_position < len(right_state), "Index out of range"
            assert left_moving_strand_position < len(right_state), "Index out of range"
            assert not right_state[
                right_moving_strand_position
            ].is_framing(), "Framing nodes can't move"

        if right_moving_strand_position >= left_moving_strand_position:
            word = tuple(
                range(left_moving_strand_position, right_moving_strand_position)
            )
        else:
            word = tuple(
                range(
                    left_moving_strand_position - 1,
                    right_moving_strand_position - 1,
                    -1,
                )
            )

        return self._element_constructor_(state=right_state, word=word)

    def braid_by_extending_permutation(
        self,
        right_state: KLRWstate,
        mapping: dict,
        check=False,
    ) -> KLRWbraid:
        """
        Returns a braid given by a permutation of (a part of) moving strands,
        the rest of strands are not intersecting.
        """

        N = max(max(mapping.values()), max(mapping.keys()))

        if check:
            assert N >= self.total_total_number_of_strands()

        right_not_moving_strands = [
            x for x in range(N + 1) if x not in mapping.values()
        ]
        left_not_moving_strands = [x for x in range(N + 1) if x not in mapping.keys()]

        # define permutation on the not moving part
        permutation = {
            b: t for b, t in zip(left_not_moving_strands, right_not_moving_strands)
        }

        # define permutation on the moving set
        for b, t in mapping.items():
            permutation[b] = t

        # make reduced word by finding elements of the Lehmer cocode
        rw = []
        for left_index in range(N + 1):
            right_index = permutation[left_index]
            # cocode for a strand indicates how many strands
            # moved through it from left to right
            cocode_element = sum(
                1 for i in range(left_index) if permutation[i] > right_index
            )
            piece = [j for j in range(left_index, left_index - cocode_element, -1)]
            rw += piece

        return self._element_constructor_(state=right_state, word=tuple(rw))

    # TODO: rewrite
    def check_minimality(self):
        pass

    # TODO: rewrite
    def check(self):
        self.check_minimality()
