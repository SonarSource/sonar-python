from typing import Any, Set, FrozenSet, Union, Coroutine, Callable, TypeVar
import asyncio

def empty_union(x: Union['A', 'B']):
  x()

def foo(param1: int, param2: Set[int], param3: FrozenSet[int], param4: list[str], param5: set[int]):
  param1() # Noncompliant {{Fix this call; Previous type checks suggest that "param1" has type int and it is not callable.}}
  x = 42
  x() # OK, raised by S5756

  param2() # Noncompliant
  s = set()
  s() # OK, raised by S5756

  param3() # Noncompliant
  fs = frozenset()
  fs() # OK, raised by S5756

  for item in param4:
      item() # Noncompliant

  param5() # Noncompliant

def derived(x: int):
  x.conjugate()() # Noncompliant
  y = x or 'str'
  y() # Noncompliant
  z = x + 42
  z() # Noncompliant

class Base: ...
class CallableBase:
  def __call__(self, *args, **kwargs): ...
def with_isinstance(x: Base):
  if isinstance(x, CallableBase):
    x()


async def bar(func: Coroutine):
  func() # it's technically possible to call a Coroutine, although it won't behave as a normal function
  await asyncio.gather(
    func()
  )

def decorators(decorator: Callable, non_callable: str):

    @non_callable()  # Noncompliant
    def foo():
        ...

    @decorator()
    def bar():
        ...

def local_function() -> int:
    ...

def calling_local_function():
    x = local_function()
    x() # Noncompliant

T = TypeVar("T", bound=Callable)
def calling_type_var(function: T):
  function()

T2 = TypeVar("T2", bound=int)
def calling_type_var(function: T2):
  function() # FN: We don't preserve the bound type of the TypeVar


def mixed_type_sources(condition, hinted: int):
    mixed = hinted if condition else 1.5
    mixed()  # Noncompliant


def inherited_callability(value: CallableBase):
    value()


from re import Pattern

def hinted_regular_expression_search_results(pattern: Pattern, text):
    match = pattern.search(text)
    if match:
        match(1)  # Noncompliant
        match.group(1)


def heterogeneous_tuple_iteration(values: tuple[int, Callable] | tuple[str, Callable]):
    for value in values:
        if callable(value):
            value()


def non_callable_tuple_iteration(values: tuple[int, str] | tuple[str, int]):
    for value in values:
        value()  # Noncompliant


def unknown_tuple_element(values: tuple[int, Any] | tuple[str, Any]):
    for value in values:
        value()


from typing import Generic, Iterator

Element = TypeVar("Element")

class Callbacks(Generic[Element]):
    def __iter__(self) -> Iterator[Callable]:
        yield lambda: None

class CallbackList(list[Element]):
    def __iter__(self) -> Iterator[Callable]:
        yield lambda: None

def custom_generic_iteration(values: Callbacks[int] | Callbacks[str]):
    for callback in values:
        callback()

def overridden_builtin_iteration(values: CallbackList[int] | CallbackList[str]):
    for callback in values:
        callback()


from typing import IO

def standard_generic_iterator_elements(stream: IO[str], filtered: filter[int], mapped: map[int], backwards: reversed[int]):
    for line in stream:
        line()  # Noncompliant
    for value in filtered:
        value()  # Noncompliant
    for value in mapped:
        value()  # Noncompliant
    for value in backwards:
        value()  # Noncompliant


from itertools import chain, islice, repeat
from weakref import WeakValueDictionary

# Accepted limitation: these iterable types are outside the explicit element-inference support set.
# Each loop variable is UNKNOWN, so S5864 intentionally leaves these invalid calls unreported.
def unlisted_standard_iterator_elements(chained: chain[int], sliced: islice[int], repeated: repeat[int], weak: WeakValueDictionary[str, object]):
    for value in chained:
        value()  # FN: Unknown element type for chain[int].
    for value in sliced:
        value()  # FN: Unknown element type for islice[int].
    for value in repeated:
        value()  # FN: Unknown element type for repeat[int].
    for key in weak:
        key()  # FN: Unknown key type for WeakValueDictionary[str, object].
