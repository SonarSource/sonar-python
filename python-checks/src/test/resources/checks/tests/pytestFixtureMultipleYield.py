import pytest


@pytest.fixture
def database_connection():
    db = create_connection()
    yield db
    yield db  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#   ^^^^^^^^
    db.close()


@pytest.fixture
def single_yield_fixture():
    resource = acquire()
    yield resource
    release(resource)


@pytest.fixture
def return_only_fixture():
    return create_connection()


def generator_with_multiple_yields():
    yield 1
    yield 2


@pytest.fixture()
def fixture_with_call_decorator():
    value = setup()
    yield value
    teardown(value)


@pytest.fixture
def fixture_with_nested_generator():
    def inner():
        yield 1
        yield 2

    inner()
    yield "ok"


@pytest.fixture
def triple_yield_fixture():
    yield 1
    yield 2  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#   ^^^^^^^
    yield 3  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#   ^^^^^^^


import unittest


class MyTest(unittest.TestCase):

    @pytest.fixture
    def class_fixture(self):
        yield "setup"
        yield "again"  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#       ^^^^^^^^^^^^^
        self.teardown()



from pytest import fixture as pytest_fixture


@pytest_fixture
def imported_fixture_decorator():
    value = setup()
    yield value
    yield value  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#   ^^^^^^^^^^^
    teardown(value)


@pytest.fixture
def exclusive_branch_yields(condition):
    if condition:
        yield "a"
    else:
        yield "b"


@pytest.fixture
def yield_then_return_on_branch(terminal_reporter):
    if terminal_reporter is None:
        yield
        return
    yield "value"


@pytest.fixture
def sequential_after_non_exiting_if(condition):
    if condition:
        yield "a"
    yield "b"  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#   ^^^^^^^^^


@pytest.fixture
def try_except_exclusive_yields():
    try:
        risky()
        yield "try-path"
    except ValueError:
        yield "except-path"


@pytest.fixture
def try_body_with_two_yields():
    try:
        yield 1
        yield 2  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#       ^^^^^^^
    except ValueError:
        pass


@pytest.fixture
def elif_chain_exclusive_yields(mode):
    if mode == 1:
        yield "first"
    elif mode == 2:
        yield "second"
    else:
        yield "third"


@pytest.fixture
def with_block_single_yield():
    with open("resource.txt") as resource:
        yield resource


@pytest.fixture
def for_loop_single_yield(items):
    for item in items:
        yield item


@pytest.fixture
def nested_return_skips_following_yield(flag):
    if flag:
        return
    yield "value"


@pytest.fixture
def while_loop_yield():
    while cond():
        yield next_item()


@pytest.fixture
def try_else_single_yield():
    try:
        work()
    except ValueError:
        pass
    else:
        yield "done"


@pytest.fixture
def try_finally_double_yield():
    try:
        yield "a"
    finally:
        yield "b"  # Noncompliant {{Pytest fixtures should contain at most one yield statement.}}
#       ^^^^^^^^^


@pytest.fixture
def try_finally_single_yield():
    try:
        work()
    finally:
        cleanup()
    yield "done"


@pytest.fixture
def nested_function_yields_ignored():
    def inner():
        yield 1
        yield 2

    yield "outer"


@pytest.fixture
def elif_branch_returns(mode):
    if mode == 1:
        pass
    elif mode == 2:
        return
    yield "value"


@pytest.fixture
def raise_exits_branch(flag):
    if flag:
        raise ValueError("stop")
    yield "ok"


@pytest.fixture
def for_with_else_yield(items):
    for item in items:
        if item:
            return
    else:
        yield "done"
