import functools

import pytest
from pytest import fixture, mark
from pytest.mark import skip


@pytest.fixture()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#              ^^
def empty_parens_fixture():
    return 1


@pytest.mark.slow()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#                ^^
def test_empty_parens_custom_mark():
    assert True


@pytest.mark.skip()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#                ^^
def test_empty_parens_known_mark():
    assert True


@pytest.mark.usefixtures()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#                       ^^
def test_empty_parens_usefixtures():
    assert True


@fixture()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#       ^^
def empty_parens_imported_fixture():
    return 1


@mark.slow()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#         ^^
def test_empty_parens_imported_mark():
    assert True


@skip()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#    ^^
def test_empty_parens_imported_skip():
    assert True


@pytest.fixture ()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#               ^^
def empty_parens_with_space_fixture():
    return 1


@pytest.mark.slow()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#                ^^
class TestEmptyParensOnClass:
    @pytest.fixture()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#                  ^^
    def resource(self):
        return 1


@pytest.mark.slow()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#                ^^
@pytest.mark.usefixtures("db")
@pytest.fixture()  # Noncompliant {{Remove empty parentheses from this decorator.}}
#              ^^
def several_decorators():
    return 1


# Compliant: no parentheses at all, which is the default expected style
@pytest.fixture
def bare_fixture():
    return 1


@pytest.mark.slow
def test_bare_custom_mark():
    assert True


@pytest.mark.skip
def test_bare_known_mark():
    assert True


@pytest.mark.usefixtures
def test_bare_usefixtures():
    assert True


@fixture
def bare_imported_fixture():
    return 1


@mark.slow
def test_bare_imported_mark():
    assert True


@pytest.mark.slow
class TestBareMarkOnClass:
    @pytest.fixture
    def resource(self):
        return 1


# Compliant: decorators with arguments keep their parentheses
@pytest.fixture(scope="module")
def scoped_fixture():
    return 1


@pytest.fixture(name="renamed", autouse=True)
def named_fixture():
    return 1


@pytest.mark.parametrize("value", [1, 2])
def test_parametrized(value):
    assert value


@pytest.mark.skip(reason="not ready")
def test_skipped():
    assert False


@pytest.mark.usefixtures("db")
def test_with_usefixtures():
    assert True


@pytest.mark.slow(1)
def test_mark_with_argument():
    assert True


def make_options():
    return {"scope": "session"}


@pytest.fixture(**make_options())
def unpacked_options_fixture():
    return 1


# Compliant: not pytest decorators
@functools.cache()
def cached():
    return 1


@functools.cache
def also_cached():
    return 1


def my_decorator(func):
    return func


@my_decorator
def decorated():
    return 1


@my_decorator()
def decorated_with_parens():
    return 1
