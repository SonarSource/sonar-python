import pytest
from pytest import fixture


# Compliant: default with no matching fixture — hard-coded optional arg, not blocked injection
def test_hardcoded_default(threshold=1):
    assert threshold > 0


def test_multiple_hardcoded(a=1, b=2):
    pass


def test_typed_hardcoded(threshold: int = 1):
    assert threshold > 0


def test_none_hardcoded(threshold=None):
    assert threshold is None


@pytest.fixture
def value():
    return 42


def test_with_default(value=1):  # Noncompliant {{Remove this default value so pytest can inject the parameter.}}
#                           ^
    assert value > 0


def test_multiple_defaults(value=1, other=2):  # Noncompliant
#                                ^
    pass


def test_typed_default(value: int = 1):  # Noncompliant
#                                   ^
    assert value > 0


def test_keyword_only(*, value=1):  # Noncompliant
#                              ^
    assert value > 0


def test_positional_only(value=1, /):  # Noncompliant
#                              ^
    assert value > 0


def test_none_default(value=None):  # Noncompliant
#                           ^^^^
    assert value is None


def test_mutable_default(value=[]):  # Noncompliant
#                              ^^
    return value


async def test_async_with_default(value=1):  # Noncompliant
#                                       ^
    assert value > 0


def test_mixed(fixture_ok, value=1):  # Noncompliant
#                                ^
    assert value > 0


# Builtin fixtures are always known — default blocks injection
def test_request_default(request=None):  # Noncompliant
#                                ^^^^
    pass


def test_monkeypatch_default(monkeypatch=None):  # Noncompliant
#                                        ^^^^
    pass


def test_tmp_path_default(tmp_path=None):  # Noncompliant
#                                  ^^^^
    pass


class TestClass:
    @pytest.fixture
    def class_value(self):
        return 1

    def test_method_with_default(self, class_value=1):  # Noncompliant
#                                                  ^
        assert class_value > 0

    def test_method_module_fixture(self, value=1):  # Noncompliant
#                                              ^
        assert value > 0

    @pytest.fixture
    def class_fixture_with_default(self, value=1):  # Noncompliant
#                                              ^
        return value


@pytest.fixture
def fixture_with_default(value=1):  # Noncompliant
#                              ^
    return value


@pytest.fixture()
def fixture_call_with_default(value=1):  # Noncompliant
#                                   ^
    return value


@fixture
def imported_fixture_with_default(value=1):  # Noncompliant
#                                       ^
    return value


@pytest.fixture
async def async_fixture_with_default(value=1):  # Noncompliant
#                                          ^
    return value


@pytest.fixture(name="renamed")
def _internal_name():
    return 1


def test_renamed_fixture(renamed=0):  # Noncompliant
#                                ^
    assert renamed >= 0


# Compliant: no defaults
def test_without_default(value):
    assert value > 0


def test_multiple_params(a, b, c):
    pass


def test_variadic(*args, **kwargs):
    pass


@pytest.fixture
def fixture_without_default(value):
    return value


@fixture()
def imported_fixture_ok(value):
    return value


class TestCompliant:
    def test_method(self, value):
        assert value > 0

    @pytest.fixture
    def class_fixture(self, value):
        return value


# Compliant: not a test or fixture
def helper_with_default(value=1):
    return value


def not_a_test(value=1):
    return value


# Compliant: test*-named method in a non-Test class is not collected by pytest
class DataBuilder:
    def test_data(self, size=10):
        return size


# Compliant: nested helper named test_* inside a non-test class
class TestS3Hook:
    def test_provide_bucket_name(self):
        class FakeS3Hook:
            def test_function(self, bucket_name=None):
                return bucket_name


# Compliant: unittest-style helpers with hardcoded defaults and no matching fixture
class ComparisonTest:
    def test_valid(self, position=None):
        pass


class DigestAuthTests:
    def test_hash(self, algorithm='md5'):
        pass


class test_DynamoDBBackend:
    def test_backend_by_url(self, url='dynamodb://'):
        pass


class RemoteFuncsTestCase:
    def test_pub_ret(self, load=None):
        pass


# Compliant: class fixture is not available outside its class
class TestOther:
    def test_no_class_fixture(self, class_value=1):
        assert class_value > 0


@pytest.mark.parametrize("value", [1, 2])
def test_parametrized_without_default(value):
    assert value > 0


# Parametrize injects the arg — default is unused / incorrect
@pytest.mark.parametrize("arg", [1, 2])
def test_parametrized_with_hardcoded(arg=0):  # Noncompliant
#                                        ^
    assert arg >= 0


@pytest.mark.parametrize("value", [1, 2])
def test_parametrized_with_default(value=0):  # Noncompliant
#                                        ^
    assert value >= 0


@pytest.mark.parametrize(("left", "right"), [(1, 2), (3, 4)])
def test_parametrized_tuple_names(left=0, right=1):  # Noncompliant 2
    pass


@pytest.mark.parametrize("a,b", [(1, 2)])
def test_parametrized_csv_names(a=0, b=1):  # Noncompliant 2
    pass


# Compliant: default on a non-parametrized, non-fixture parameter alongside parametrize
@pytest.mark.parametrize("case", [1, 2])
def test_parametrized_with_unrelated_default(case, threshold=10):
    assert case < threshold
