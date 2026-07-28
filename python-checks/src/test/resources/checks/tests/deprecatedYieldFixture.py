import pytest


@pytest.yield_fixture  # Noncompliant {{Replace deprecated pytest.yield_fixture with pytest.fixture.}}
#^^^^^^^^^^^^^^^^^^^^
def bare_yield_fixture():
    value = acquire()
    yield value
    release(value)


@pytest.yield_fixture()  # Noncompliant {{Replace deprecated pytest.yield_fixture with pytest.fixture.}}
#^^^^^^^^^^^^^^^^^^^^^^
def called_yield_fixture():
    value = acquire()
    yield value
    release(value)


@pytest.yield_fixture(scope="module")  # Noncompliant {{Replace deprecated pytest.yield_fixture with pytest.fixture.}}
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def yield_fixture_with_kwargs():
    value = acquire()
    yield value
    release(value)


from pytest import yield_fixture


@yield_fixture  # Noncompliant {{Replace deprecated pytest.yield_fixture with pytest.fixture.}}
#^^^^^^^^^^^^^
def imported_yield_fixture():
    value = acquire()
    yield value
    release(value)


from pytest import yield_fixture as legacy_fixture


@legacy_fixture  # Noncompliant {{Replace deprecated pytest.yield_fixture with pytest.fixture.}}
#^^^^^^^^^^^^^^
def aliased_imported_yield_fixture():
    value = acquire()
    yield value
    release(value)


import pytest as pt


@pt.yield_fixture  # Noncompliant {{Replace deprecated pytest.yield_fixture with pytest.fixture.}}
#^^^^^^^^^^^^^^^^
def aliased_module_yield_fixture():
    value = acquire()
    yield value
    release(value)


@pytest.fixture
def compliant_fixture():
    value = acquire()
    yield value
    release(value)


@pytest.fixture()
def compliant_called_fixture():
    value = acquire()
    yield value
    release(value)


@pytest.fixture(scope="module")
def compliant_fixture_with_kwargs():
    value = acquire()
    yield value
    release(value)


class NotPytest:
    @staticmethod
    def yield_fixture(*args, **kwargs):
        pass


@NotPytest.yield_fixture
def decoy_class_attribute():
    pass


@bob.bob
def decoy_qualified_expression_null_symbol():
    pass


@"my string"
def decoy_decorator_without_symbol():
    pass
