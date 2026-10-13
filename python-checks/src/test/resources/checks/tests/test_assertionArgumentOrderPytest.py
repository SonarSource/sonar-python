import pytest

EXPECTED_COUNT = 42
EXPECTED_PI = 3.14


def value():
    return 41 + 1


# "actual first" is used 6 times and "expected first" 5 times, so the latter deviates from the file convention
def test_pytest_assertions():
    assert 42 == value()
#   Noncompliant@-1 {{Swap these 2 operands so the actual value comes first, to match the rest of this file.}}
#          ^^^^^^^^^^^^^@-1
#          ^^@-2< {{Expected value.}}
#                ^^^^^^^@-3< {{Actual value.}}
    assert EXPECTED_COUNT == value()  # Noncompliant
    assert pytest.approx(EXPECTED_PI) == value()  # Noncompliant
    assert 42 == pytest.approx(value())
#   Noncompliant@-1
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1
#          ^^@-2< {{Expected value.}}
#                              ^^^^^^^@-3< {{Actual value.}}
    assert 42 == pytest.approx(other=value())  # Noncompliant

    assert value() == 42
    assert value() == EXPECTED_COUNT
    assert value() == pytest.approx(EXPECTED_PI)
    assert value() == None
    assert value() == "ok"
    assert value() == 3.14
    assert value() == value()
    assert 42 == EXPECTED_COUNT
    assert value()
    assert 42 != value()
