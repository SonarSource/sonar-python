import contextlib
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Iterator
import some_unknown_module


class OnlyEnter:
    def __enter__(self):
        return self


class OnlyExit:
    def __exit__(self, exc_type, exc_value, traceback):
        return False


class NotCM:
    pass


class SyncCM:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class SubCM(SyncCM):
    pass


class AsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class OnlyAenter:
    async def __aenter__(self):
        return self


class DualCM:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


@contextmanager
def make_cm():
    yield 1


def sync_noncompliant():
    with OnlyEnter() as r:  # Noncompliant {{Change this expression to a context manager (implementing "__enter__" and "__exit__").}}
#        ^^^^^^^^^^^
        print(r)
    with OnlyExit():  # Noncompliant
#        ^^^^^^^^^^
        pass
    with NotCM():  # Noncompliant
#        ^^^^^^^
        pass
    with 5:  # Noncompliant
#        ^
        pass
    with "abc":  # Noncompliant
#        ^^^^^
        pass
    with [1, 2]:  # Noncompliant
#        ^^^^^^
        pass
    with None:  # Noncompliant
#        ^^^^
        pass
    with SyncCM(), NotCM():  # Noncompliant
#                  ^^^^^^^
        pass


def sync_compliant():
    with SyncCM() as c:  # Compliant
        print(c)
    with SubCM() as c:  # Compliant: protocol inherited from SyncCM
        print(c)
    with open("file.txt") as f:  # Compliant: file object is a context manager
        print(f)
    with make_cm() as value:  # Compliant: @contextmanager builds a context manager
        print(value)


def sync_unknown_no_fp(param):
    with param as x:  # Compliant: parameter type is unknown
        print(x)
    with some_unknown_module.get() as y:  # Compliant: unresolved import
        print(y)


def sync_union_no_fp(flag):
    cm = SyncCM() if flag else None
    with cm as x:  # Compliant: one union member (SyncCM) is a valid context manager
        print(x)


def sync_union_all_bad(flag):
    value = 5 if flag else "abc"
    with value:  # Noncompliant
#        ^^^^^
        pass


async def async_noncompliant():
    async with SyncCM() as c:  # Noncompliant {{Change this expression to an asynchronous context manager (implementing "__aenter__" and "__aexit__").}}
#              ^^^^^^^^
        print(c)
    async with OnlyAenter():  # Noncompliant
#              ^^^^^^^^^^^^
        pass
    async with NotCM():  # Noncompliant
#              ^^^^^^^
        pass


async def async_compliant():
    async with AsyncCM() as c:  # Compliant
        print(c)
    async with DualCM() as c:  # Compliant: implements both protocols
        print(c)


async def async_unknown_no_fp(param):
    async with param as x:  # Compliant: unknown type
        print(x)


async def defer_to_s7515():
    # Inside an async function, an async context manager used with a plain "with" is S7515's responsibility (it recommends "async with").
    with AsyncCM() as c:  # Compliant
        print(c)
    with DualCM() as c:  # Compliant: valid synchronous context manager
        print(c)
    # A plain non-context-manager is still reported inside an async function: S7515 only covers async managers.
    with 5:  # Noncompliant
#        ^
        pass


def async_only_in_sync_with():
    # In a synchronous function "async with" is not an option, so S7515 never fires here: an async-only
    # object used in a plain "with" fails at runtime and must be reported by this rule.
    with AsyncCM() as c:  # Noncompliant
#        ^^^^^^^^^
        print(c)


# ----- Several context managers in a single statement are checked independently -----
def multiple_items():
    with SyncCM() as a, DualCM() as b:  # Compliant: both are valid context managers
        print(a, b)
    with NotCM(), SyncCM():  # Noncompliant
#        ^^^^^^^
        pass
    with SyncCM(), OnlyEnter():  # Noncompliant
#                  ^^^^^^^^^^^
        pass


def multiple_items_parenthesized():
    with (
        SyncCM(),
        NotCM(),  # Noncompliant
        OnlyExit(),  # Noncompliant
    ):
        pass


# ----- A plain generator or iterator is not a context manager -----
def gen():
    yield 1


def annotated_gen() -> Generator[int, None, None]:  # no @contextmanager: a raw generator, not a context manager
    yield 1


def generators_are_not_context_managers():
    # A generator function without @contextmanager, and a generator expression, are not context managers.
    with annotated_gen():  # Noncompliant
        pass
    with (x for x in range(3)):  # Noncompliant
        pass
    # Accepted FN: an unannotated generator function has an unresolved return type, so it cannot be reported.
    with gen():  # Compliant (FN)
        pass


# ----- Variables and annotated parameters carry their inferred type -----
def variables():
    bad = NotCM()
    with bad:  # Noncompliant
#        ^^^
        pass
    good = SyncCM()
    with good:  # Compliant
        print(good)


def annotated_parameters(bad: int, good: SyncCM):
    with bad:  # Noncompliant
#        ^^^
        pass
    with good:  # Compliant
        print(good)


# ----- A function object is not a context manager -----
def plain_function():
    pass


def function_object_used_as_context_manager():
    # Accepted FN: the function type does not report a definite "no __enter__", so it is not flagged.
    with plain_function:  # Compliant (FN)
        pass


# ----- Nested statements are checked independently -----
def nested():
    with SyncCM():  # Compliant
        with NotCM():  # Noncompliant
            pass


# ----- The "as" target shape is irrelevant -----
def as_tuple_target():
    with SyncCM() as (a, b):  # Compliant
        print(a, b)


# ----- contextlib helpers are context managers -----
def contextlib_helpers():
    with contextlib.suppress(ValueError):  # Compliant
        pass
    with contextlib.ExitStack() as stack:  # Compliant
        print(stack)


# ----- async: several context managers in a single statement -----
async def async_multiple_items():
    async with AsyncCM(), DualCM():  # Compliant
        pass
    async with AsyncCM(), SyncCM():  # Noncompliant
#                         ^^^^^^^^
        pass


# ----- Special methods assigned as instance attributes do not satisfy the protocol -----
class InstanceAttributeCM:
    def __init__(self):
        self.__enter__ = lambda: self
        self.__exit__ = lambda *args: False


def instance_attribute_dunders():
    with InstanceAttributeCM():  # Noncompliant
        pass


# ----- @contextmanager / @asynccontextmanager functions are inferred as their underlying generator -----
# The decorator is not modeled by type inference, so the value is a Generator/Iterator/Coroutine. These are the
# common real-world shape (e.g. Airflow's "create_session") and must not be flagged.
@contextmanager
def annotated_session() -> Generator[int, None, None]:
    yield 1


@contextmanager
def annotated_session_iter() -> Iterator[int]:
    yield 1


@asynccontextmanager
async def annotated_async_session() -> AsyncGenerator[int, None]:
    yield 1


@asynccontextmanager
async def unannotated_async_session():
    yield 1


def contextmanager_factories():
    with annotated_session() as value:  # Compliant
        print(value)
    with annotated_session_iter() as value:  # Compliant
        print(value)


async def asynccontextmanager_factories():
    async with annotated_async_session() as value:  # Compliant
        print(value)
    async with unannotated_async_session() as value:  # Compliant
        print(value)


def contextmanager_result_stored_or_conditional(flag):
    # The result of a @contextmanager is inferred as a plain generator. When it does not come straight from a call
    # we can inspect (stored in a variable, produced by a conditional), we trust the generator type to avoid a FP.
    session = annotated_session()
    with session as value:  # Compliant
        print(value)
    with (annotated_session() if flag else annotated_session_iter()) as value:  # Compliant
        print(value)
    # Accepted FN: a raw generator stored before use cannot be distinguished from a @contextmanager result.
    raw = annotated_gen()
    with raw:  # Compliant (FN)
        pass


async def asynccontextmanager_result_stored():
    session = annotated_async_session()
    async with session as value:  # Compliant
        print(value)


# ----- @contextmanager methods called via attribute access -----
class DatabaseHelper:
    @contextmanager
    def session_scope(self) -> Generator[int, None, None]:
        yield 1

    @asynccontextmanager
    async def async_session_scope(self):
        yield 1

    def run(self):
        with self.session_scope() as session:  # Compliant
            print(session)

    async def run_async(self):
        async with self.async_session_scope() as session:  # Compliant
            print(session)
