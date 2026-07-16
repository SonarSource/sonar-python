import pytest
from pytest import fixture


@pytest.fixture(autouse=True, params=[1, 2, 3])  # Noncompliant {{Remove the "params" argument or set "autouse" to False.}}
def noncompliant_module_fixture():
    return "value"


@pytest.fixture(params=[1, 2, 3], autouse=True)  # Noncompliant
def noncompliant_reordered_arguments():
    return "value"


@pytest.fixture(autouse=True)
def compliant_autouse_only():
    return "value"


@pytest.fixture(params=[1, 2, 3])
def compliant_params_only(request):
    return request.param


@pytest.fixture(autouse=False, params=[1, 2, 3])
def compliant_autouse_false_with_params():
    return "value"


@pytest.fixture(autouse=True, params=[])
def compliant_empty_params():
    return "value"


@pytest.fixture(autouse=True, params=None)
def compliant_none_params():
    return "value"


@pytest.fixture
def compliant_no_arguments():
    return "value"


class TestSomething:
    @pytest.fixture(autouse=True, params=[1, 2])  # Noncompliant
    def noncompliant_class_fixture(self):
        pass


@fixture(autouse=True, params=[1])  # Noncompliant
def noncompliant_imported_fixture():
    return "value"


PARAMS = [1, 2]
@pytest.fixture(autouse=True, params=PARAMS)  # Noncompliant
def noncompliant_params_from_variable():
    return "value"
