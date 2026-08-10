import pytest


def test_file_operation(tmp_path_factory):  # Noncompliant {{Use "tmp_path" instead of "tmp_path_factory" in function-scoped tests.}}
#                       ^^^^^^^^^^^^^^^^
    temp_dir = tmp_path_factory.mktemp('data')
    test_file = temp_dir / 'test.txt'
    test_file.write_text('content')
    assert test_file.read_text() == 'content'


def test_with_other_fixtures(tmp_path_factory, monkeypatch):  # Noncompliant
#                            ^^^^^^^^^^^^^^^^
    temp_dir = tmp_path_factory.mktemp('data')
    assert temp_dir.exists()


async def test_async(tmp_path_factory):  # Noncompliant
#                    ^^^^^^^^^^^^^^^^
    temp_dir = tmp_path_factory.mktemp('data')
    assert temp_dir.exists()


@pytest.mark.parametrize("value", [1, 2])
def test_parametrized(tmp_path_factory, value):  # Noncompliant
#                     ^^^^^^^^^^^^^^^^
    assert (tmp_path_factory.mktemp('data') / str(value)).parent.exists()


class TestSomething:
    def test_method(self, tmp_path_factory):  # Noncompliant
#                         ^^^^^^^^^^^^^^^^
        temp_dir = tmp_path_factory.mktemp('data')
        assert temp_dir.exists()


def test_file_operation_compliant(tmp_path):
    test_file = tmp_path / 'test.txt'
    test_file.write_text('content')
    assert test_file.read_text() == 'content'


def test_no_tmp_path():
    assert True


@pytest.fixture
def function_scoped_fixture(tmp_path_factory):
    return tmp_path_factory.mktemp('data')


@pytest.fixture(scope="session")
def session_data(tmp_path_factory):
    return tmp_path_factory.mktemp('session')


@pytest.fixture(scope="module")
def module_data(tmp_path_factory):
    return tmp_path_factory.mktemp('module')


@pytest.fixture(scope="class")
def class_data(tmp_path_factory):
    return tmp_path_factory.mktemp('class')


@pytest.fixture(scope="package")
def package_data(tmp_path_factory):
    return tmp_path_factory.mktemp('package')


def helper_not_a_test(tmp_path_factory):
    return tmp_path_factory.mktemp('helper')


def setup_module(tmp_path_factory):
    return tmp_path_factory.mktemp('setup')


class TestHelpers:
    def helper_method(self, tmp_path_factory):
        return tmp_path_factory.mktemp('helper')

    @pytest.fixture
    def test_looking_fixture(self, tmp_path_factory):
        return tmp_path_factory.mktemp('fixture')
