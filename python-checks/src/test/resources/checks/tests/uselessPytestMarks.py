import pytest
from pytest import fixture, mark
from pytest.mark import usefixtures, skip


@pytest.mark.asyncio  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=20]
@pytest.fixture
async def db_async_mark():
    return await connect()


@pytest.mark.usefixtures("db")  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=30]
@pytest.fixture
def client_usefixtures_on_fixture(db):
    return Client(db)


@pytest.mark.slow  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=17]
@pytest.fixture
def cache_custom_mark():
    return {}


@pytest.mark.parametrize("x", [1, 2])  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=37]
@pytest.fixture
def parametrized_fixture(x):
    return x


@pytest.mark.skip  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=17]
@pytest.fixture
def skipped_fixture():
    return 1


@pytest.fixture
@pytest.mark.asyncio  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=20]
async def mark_below_fixture():
    return 1


@pytest.mark.slow  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=17]
@pytest.mark.usefixtures("db")  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=30]
@pytest.fixture
def multiple_marks_on_fixture():
    return 1


@mark.slow  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=10]
@fixture
def imported_mark_on_fixture():
    return 1


@pytest.fixture
@skip  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=5]
def imported_skip_on_fixture():
    return 1


@pytest.mark.usefixtures()  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=26]
@pytest.fixture
def empty_usefixtures_on_fixture():
    return 1


@pytest.mark.usefixtures  # Noncompliant {{Remove this mark; it has no effect on fixtures.}}
#^[sc=1;ec=24]
@pytest.fixture
def bare_usefixtures_on_fixture():
    return 1


@pytest.mark.usefixtures()  # Noncompliant {{Provide fixture names or remove this empty usefixtures decorator.}}
#^[sc=1;ec=26]
def test_empty_usefixtures():
    assert ping() == "pong"


@pytest.mark.usefixtures()  # Noncompliant {{Provide fixture names or remove this empty usefixtures decorator.}}
#^[sc=1;ec=26]
class TestEmptyUsefixtures:
    def test_something(self):
        assert True


@mark.usefixtures()  # Noncompliant {{Provide fixture names or remove this empty usefixtures decorator.}}
#^[sc=1;ec=19]
def test_empty_usefixtures_via_mark_import():
    assert True


@usefixtures()  # Noncompliant {{Provide fixture names or remove this empty usefixtures decorator.}}
#^[sc=1;ec=14]
def test_empty_usefixtures_direct_import():
    assert True


# Compliant: clean fixtures
@pytest.fixture
async def db():
    return await connect()


@pytest.fixture
def client(db):
    return Client(db)


@fixture
def cache():
    return {}


# Compliant: marks on tests / classes (not fixtures)
@pytest.mark.asyncio
async def test_async():
    assert True


@pytest.mark.slow
def test_slow():
    assert True


@pytest.mark.usefixtures("db")
def test_with_usefixtures():
    assert True


@pytest.mark.usefixtures
def test_bare_usefixtures():
    assert True


@pytest.mark.usefixtures("db", "cache")
class TestWithUsefixtures:
    def test_something(self):
        assert True


@mark.parametrize("x", [1])
def test_parametrized(x):
    assert x == 1


@pytest.mark.skip(reason="not ready")
def test_skipped():
    assert False


def not_a_test():
    pass


@pytest.mark.slow
def helper_not_fixture():
    return 1
