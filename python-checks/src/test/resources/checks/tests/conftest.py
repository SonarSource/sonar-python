import pytest


@pytest.fixture
def database():
    return object()


@pytest.fixture
def combined(request):
    return request.getfixturevalue('database')


def pytest_configure(config):
    request = config
    request.getfixturevalue('database')


def test_in_conftest_should_be_ignored(request):
    # Tests in conftest.py are not collected; still exercise the conftest exclusion path.
    request.getfixturevalue('database')
