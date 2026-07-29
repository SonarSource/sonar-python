import functools

import pytest
from pytest import fixture, mark
from pytest.mark import skip


@pytest.fixture  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^^^^^^^^^^^
def bare_fixture():
    return 1


@pytest.mark.slow  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^^^^^^^^^^^^^
def test_bare_custom_mark():
    assert True


@pytest.mark.skip  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^^^^^^^^^^^^^
def test_bare_known_mark():
    assert True


@pytest.mark.usefixtures  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^^^^^^^^^^^^^^^^^^^^
def test_bare_usefixtures():
    assert True


@fixture  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^^^^
def bare_imported_fixture():
    return 1


@mark.slow  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^^^^^^
def test_bare_imported_mark():
    assert True


@skip  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^
def test_bare_imported_skip():
    assert True


@pytest.mark.slow  # Noncompliant {{Add empty parentheses to this decorator.}}
#^^^^^^^^^^^^^^^^
class TestBareMarkOnClass:
    @pytest.fixture  # Noncompliant {{Add empty parentheses to this decorator.}}
#    ^^^^^^^^^^^^^^
    def resource(self):
        return 1


# Compliant: parentheses are present, which is the configured expected style
@pytest.fixture()
def empty_parens_fixture():
    return 1


@pytest.mark.slow()
def test_empty_parens_custom_mark():
    assert True


@fixture()
def empty_parens_imported_fixture():
    return 1


@mark.slow()
def test_empty_parens_imported_mark():
    assert True


# Compliant: decorators with arguments already have parentheses
@pytest.fixture(scope="module")
def scoped_fixture():
    return 1


@pytest.mark.parametrize("value", [1, 2])
def test_parametrized(value):
    assert value


@pytest.mark.usefixtures("db")
def test_with_usefixtures():
    assert True


# Compliant: not pytest decorators
@functools.cache
def cached():
    return 1


def my_decorator(func):
    return func


@my_decorator
def decorated():
    return 1
