import pytest


@pytest.fixture
def useless_yield():
    data = load()
    yield data  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^^^


@pytest.fixture()
def useless_yield_with_call():
    resource = acquire()
    yield resource  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^^^^^^^


@pytest.fixture
def bare_yield_only():
    yield  # Noncompliant {{Remove this useless "yield".}}
#   ^^^^^


@pytest.fixture(autouse=True)
def clear_db():
    clear_db_logs()
    clear_db_runs()
    clear_db_dags()
    yield  # Noncompliant {{Remove this useless "yield".}}
#   ^^^^^


@pytest.fixture
def only_yield_value():
    yield 42  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^


from pytest import fixture


@fixture
def imported_fixture():
    yield load()  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^^^^^


from pytest import fixture as pytest_fixture


@pytest_fixture
def aliased_fixture():
    value = setup()
    yield value  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^^^^


@pytest.fixture(scope="module")
def scoped_useless_yield():
    yield create()  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^^^^^^^


@pytest.fixture
def with_teardown():
    resource = acquire()
    yield resource
    resource.release()


@pytest.fixture
def with_pass_after_yield():
    yield 1
    pass


@pytest.fixture
def bare_yield_with_teardown():
    setup()
    yield
    teardown()


@pytest.fixture
def return_fixture():
    return load()


@pytest.fixture
def return_after_setup():
    data = load()
    return data


def not_a_fixture():
    yield 1


@pytest.fixture
def nested_generator_ok():
    def inner():
        yield 1

    return inner()


@pytest.fixture
def nested_yield_then_yield():
    def inner():
        yield 1

    yield inner()  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^^^^^^


@pytest.fixture
def multiple_yields_last_is_yield():
    # Covered by S8994; do not suggest return/remove on the last yield alone.
    yield 1
    yield 2


@pytest.fixture
def yield_in_if_not_last():
    if cond:
        yield 1
    else:
        yield 2


@pytest.fixture
async def async_useless_yield():
    yield await load()  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#   ^^^^^^^^^^^^^^^^^^


import unittest
from abc import ABC, abstractmethod


class MyTest(unittest.TestCase):

    @pytest.fixture
    def class_fixture(self):
        yield "setup"  # Noncompliant {{No teardown in this fixture. Use "return" instead of "yield".}}
#       ^^^^^^^^^^^^^

    @pytest.fixture
    def class_fixture_with_teardown(self):
        yield "setup"
        self.cleanup()


class AbstractFixtures(ABC):

    @pytest.fixture
    @abstractmethod
    def abstract_fixture(self):
        yield 1


def _restore_logging():
    handlers = []
    try:
        yield
    finally:
        pass


@pytest.fixture
def restore_logging_via_yield_from():
    yield from _restore_logging()


@pytest.fixture
def useless_yield_from_without_teardown():
    def values():
        yield 1

    yield from values()
