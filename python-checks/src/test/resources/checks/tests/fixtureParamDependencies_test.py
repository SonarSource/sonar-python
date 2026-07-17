import pytest


@pytest.fixture
def database():
    class DB:
        def query(self, sql):
            return 1

    return DB()


@pytest.fixture
def cache():
    return object()


def test_database_query(request):
    db = request.getfixturevalue('database')  # Noncompliant {{Declare this fixture as a test function parameter instead of using "request.getfixturevalue()" with a string literal.}}
#                                ^^^^^^^^^^
    result = db.query('SELECT 1')
    assert result == 1


def test_database_query_keyword(request):
    db = request.getfixturevalue(argname='database')  # Noncompliant {{Declare this fixture as a test function parameter instead of using "request.getfixturevalue()" with a string literal.}}
    assert db is not None


def test_database_query_compliant(database):
    result = database.query('SELECT 1')
    assert result == 1


@pytest.fixture(scope="module")
def combined_with_scope(request):
    return request.getfixturevalue('database')


@pytest.fixture
def combined(request):
    return request.getfixturevalue('database')


def test_no_fixture_name_argument(request):
    request.getfixturevalue()


def test_other_calls(request, database):
    database.query('SELECT 1')


def test_other_fixture_request_method(request):
    request.addfinalizer('cleanup')


@pytest.mark.parametrize('fixture_name', ['database', 'cache'])
def test_matrix(request, fixture_name):
    fixture = request.getfixturevalue(fixture_name)
    assert fixture is not None


def test_dynamic_fixture_name(request):
    name = 'database'
    fixture = request.getfixturevalue(name)
    assert fixture is not None


def helper_not_a_test(request):
    request.getfixturevalue('database')


class TestClass:
    def test_in_class(self, request):
        request.getfixturevalue('database')  # Noncompliant {{Declare this fixture as a test function parameter instead of using "request.getfixturevalue()" with a string literal.}}
