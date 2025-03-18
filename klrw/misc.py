from typing import Callable, Iterable


def get_from_all_and_assert_equality(func: Callable, objects: Iterable):
    """
    Return `func(x)` for `x`in `objects`.
    Checking if it's the same for all `x`
    """
    iterator = iter(objects)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Need at least one element")
    result = func(first)
    assert all(func(obj) == result for obj in objects), "Objects do not match"

    return result
