import pytest
from pytest import fixture


@pytest.fixture("module")  # Noncompliant {{Pass fixture options as keyword arguments.}}
#               ^^^^^^^^
def noncompliant_positional_scope():
    return []


@pytest.fixture(True)  # Noncompliant
#               ^^^^
def noncompliant_positional_autouse():
    return []


@pytest.fixture("module", True)  # Noncompliant
#               ^^^^^^^^
def noncompliant_multiple_positionals():
    return []


@pytest.fixture("module", autouse=True)  # Noncompliant
#               ^^^^^^^^
def noncompliant_mixed_positional_and_keyword():
    return []


OPTS = ("module",)
@pytest.fixture(*OPTS)  # Noncompliant
#               ^^^^^
def noncompliant_unpacked_positional():
    return []


@fixture("session")  # Noncompliant
#        ^^^^^^^^^
def noncompliant_imported_fixture():
    return []


class TestSomething:
    @pytest.fixture("class")  # Noncompliant
    #               ^^^^^^^
    def noncompliant_class_fixture(self):
        return []


@pytest.fixture(scope="module")
def compliant_keyword_scope():
    return []


@pytest.fixture(autouse=True)
def compliant_keyword_autouse():
    return []


@pytest.fixture(scope="module", autouse=True)
def compliant_multiple_keywords():
    return []


@pytest.fixture()
def compliant_empty_call():
    return []


@pytest.fixture
def compliant_no_parentheses():
    return []


@pytest.fixture(**{"scope": "module"})
def compliant_unpacked_keywords():
    return []


@fixture(scope="function")
def compliant_imported_fixture():
    return []


def not_a_fixture(*args):
    pass


@not_a_fixture("module")
def decorated_with_unrelated_call():
    pass
