import pytest

EXPECTED_COUNT = 42
EXPECTED_PI = 3.14


def value():
    return 41 + 1


def test_pytest_assertions():
    assert 42 == value()
#   Noncompliant@-1 {{Unify assertion argument order in this file; both "actual first" and "expected first" conventions are used.}}
#          ^^^^^^^^^^^^^@-1
#          ^^^^^^^^^^^^^@-2< {{Expected value first.}}
    assert EXPECTED_COUNT == value()
#          ^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
    assert pytest.approx(EXPECTED_PI) == value()
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}

    assert value() == 42
#          ^^^^^^^^^^^^^< {{Actual value first.}}
    assert value() == EXPECTED_COUNT
#          ^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
    assert value() == pytest.approx(EXPECTED_PI)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Actual value first.}}
    assert value() == value()
    assert 42 == EXPECTED_COUNT
    assert value()
    assert 42 != value()
    assert 42 == pytest.approx(value())
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
    assert 42 == pytest.approx(other=value())
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{Expected value first.}}
