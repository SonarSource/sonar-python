import pytest
from pytest import fixture


@pytest.fixture(scope="function")  # Noncompliant {{Remove this redundant scope="function" argument.}}
#               ^^^^^^^^^^^^^^^^
def noncompliant_default_scope():
    return {}


@pytest.fixture(scope='function')  # Noncompliant
#               ^^^^^^^^^^^^^^^^
def noncompliant_single_quotes():
    return {}


@pytest.fixture(scope="function", autouse=True)  # Noncompliant
#               ^^^^^^^^^^^^^^^^
def noncompliant_with_other_args():
    return {}


@pytest.fixture(autouse=True, scope="function")  # Noncompliant
#                             ^^^^^^^^^^^^^^^^
def noncompliant_scope_last():
    return {}


@fixture(scope="function")  # Noncompliant
#        ^^^^^^^^^^^^^^^^
def noncompliant_imported_fixture():
    return {}


class TestSomething:
    @pytest.fixture(scope="function")  # Noncompliant
    #               ^^^^^^^^^^^^^^^^
    def noncompliant_class_fixture(self):
        return {}


@pytest.fixture(
    scope="function",  # Noncompliant
#   ^^^^^^^^^^^^^^^^
)
def noncompliant_multiline():
    return {}


@pytest.fixture
def compliant_no_args():
    return {}


@pytest.fixture()
def compliant_empty_parens():
    return {}


@pytest.fixture(scope="module")
def compliant_module_scope():
    return {}


@pytest.fixture(scope="class")
def compliant_class_scope():
    return {}


@pytest.fixture(scope="session")
def compliant_session_scope():
    return {}


@pytest.fixture(scope="package")
def compliant_package_scope():
    return {}


@pytest.fixture(autouse=True)
def compliant_autouse_only():
    return {}


@pytest.fixture(name="renamed")
def compliant_other_options():
    return {}


SCOPE = "function"


@pytest.fixture(scope=SCOPE)
def compliant_scope_from_variable():
    return {}


def not_a_fixture(scope="function"):
    return scope
