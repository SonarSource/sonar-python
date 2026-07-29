import pytest
from pytest import fail
from pytest.mark import parametrize


def test_assert_false_with_multiple_pytest_imports():
    assert False  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^^^^^
    fail("ok")


@parametrize("n", [1])
def test_parametrize_import_enables_assert(n):
    assert False  # Noncompliant {{Replace this assertion with pytest.fail(...) and provide a message.}}
#   ^^^^^^^^^^^^
