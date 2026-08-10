import pytest
from pytest import fixture


@pytest.fixture
def dep():
    return 0


# Fixtures are flagged even in conftest.py (not a test_*.py / *_test.py file name)
@pytest.fixture
def fixture_with_default(dep=1):  # Noncompliant {{Remove this default value so pytest can inject the parameter.}}
#                            ^
    return dep


@fixture
def imported_fixture_with_default(dep=1):  # Noncompliant
#                                     ^
    return dep


@pytest.fixture
def fixture_ok(dep):
    return dep


# Compliant: optional param with no matching fixture
@pytest.fixture
def fixture_optional_loader(loader=None):
    return loader


# Module-level test_* in conftest is not a pytest-collected test file pattern for our util,
# so this should not be flagged (conftest.py is not test_*.py / *_test.py)
def test_not_collected_in_conftest(value=1):
    assert value > 0
