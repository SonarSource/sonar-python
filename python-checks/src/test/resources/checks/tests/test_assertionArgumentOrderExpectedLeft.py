import pytest

EXPECTED_COUNT = 42
EXPECTED_PI = 3.14


def value():
    return 41 + 1


# Consistent actual-first convention → no issue
def test_pytest_assertions_actual_first():
    assert value() == 42
    assert value() == EXPECTED_COUNT
    assert value() == pytest.approx(EXPECTED_PI)
    assert 42 == EXPECTED_COUNT
    assert value() == value()
