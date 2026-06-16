"""Synthetic analyzer fixture with deliberately broken rules."""

from calendar import day_abbr


# TODO (@alice) replace this placeholder path handling.
# FIXME (@bob) tighten this temporary exception flow.


def raise_generic_exception() -> None:
    """Trigger a generic exception issue."""
    raise Exception("generic failure")


def repeated_literal_value() -> str:
    """Trigger a duplicated string literal issue."""
    first = "alpha beta"
    second = "alpha beta"
    third = "alpha beta"
    return first + second + third


def unreachable_except_example() -> int:
    """Trigger an unreachable except block issue."""
    try:
        raise TypeError("bad type")
    except TypeError:
        handled = 1
        return handled
    except TypeError:
        return 2


def collapsible_if_example(first_flag: bool, second_flag: bool) -> int:
    """Trigger a collapsible nested if issue."""
    result = 0
    if first_flag:
        if second_flag:
            result = 1
    return result


class MyClass_WithNotCompliantName1:
    """Trigger a class naming issue."""

    def value(self) -> int:
        """Return a fixed value."""
        return 1


def Badly_Named_Function(value: int) -> int:
    """Trigger a function naming issue."""
    return value


def ignored_parameter_example(amount: int) -> int:
    """Trigger an ignored parameter issue."""
    amount = 7
    return amount


def bad_parameter_name(inputPar: int) -> int:
    """Trigger a parameter naming issue."""
    return inputPar + 1


def float_equality_check(value: float) -> bool:
    """Trigger a floating point equality issue."""
    return value == 0.1


def unused_local_example(value: int) -> int:
    """Trigger an unused local variable issue."""
    unread_local = value + 1
    return value


def self_assignment_example(value: int) -> int:
    """Trigger a self-assignment issue."""
    same_value = value
    same_value = same_value
    return same_value


def empty_nested_block_example(flag: bool) -> int:
    """Trigger an empty nested block issue."""
    if flag:
        pass
    return 0
