from pytest import fixture


@fixture
def combined(request):
    return request.getfixturevalue('database')


def pytest_runtest_setup(request):
    request.getfixturevalue('database')


def test_with_imported_fixture_decorator(request):
    request.getfixturevalue('database')  # Noncompliant {{Declare this fixture as a test function parameter instead of using "request.getfixturevalue()" with a string literal.}}
#                           ^^^^^^^^^^
