"""Synthetic analyzer fixture with deliberately broken rules."""

import math
from calendar import day_abbr


def raise_generic_exception() -> None:
    """Trigger a generic exception issue."""
    raise RuntimeError("generic failure")


ALPHA_BETA = "alpha beta"


def repeated_literal_value() -> str:
    """Trigger a duplicated string literal issue."""
    first = ALPHA_BETA
    second = ALPHA_BETA
    third = ALPHA_BETA
    return first + second + third


def unreachable_except_example() -> int:
    """Trigger an unreachable except block issue."""
    try:
        raise TypeError("bad type")
    except TypeError:
        handled = 1
        return handled


def collapsible_if_example(first_flag: bool, second_flag: bool) -> int:
    """Trigger a collapsible nested if issue."""
    result = 0
    if first_flag and second_flag:
        result = 1
    return result


class MyClassWithCompliantName1:
    """Trigger a class naming issue."""

    def value(self) -> int:
        """Return a fixed value."""
        return 1


def badly_named_function(value: int) -> int:
    """Trigger a function naming issue."""
    return value


def ignored_parameter_example(amount: int) -> int:
    """Trigger an ignored parameter issue."""
    adjusted_amount = 7
    return adjusted_amount


def bad_parameter_name(input_par: int) -> int:
    """Trigger a parameter naming issue."""
    return input_par + 1


def float_equality_check(value: float) -> bool:
    """Trigger a floating point equality issue."""
    return math.isclose(value, 0.1)


def unused_local_example(value: int) -> int:
    """Trigger an unused local variable issue."""
    return value


def self_assignment_example(value: int) -> int:
    """Trigger a self-assignment issue."""
    same_value = value
    return same_value


def empty_nested_block_example(flag: bool) -> int:
    """Trigger an empty nested block issue."""
    return 0
