from typing import Set, FrozenSet, Union, Coroutine, Callable
import asyncio

def empty_union(x: Union['A', 'B']):
  x()

def foo(param1: int, param2: Set[int], param3: FrozenSet[int], param4: list[str], param5: set[int]):
  param1() # Noncompliant {{Fix this call; Previous type checks suggest that "param1" has type int and it is not callable.}}
  x = 42
  x() # OK, raised by S5756

  param2() # FN SONARPY-2022
  s = set()
  s() # OK, raised by S5756

  param3() # FN SONARPY-2022
  fs = frozenset()
  fs() # OK, raised by S5756

  for item in param4:
      item() # Noncompliant

  param5() # Noncompliant

def derived(x: int):
  x.conjugate()() # FN need return type's type source resolution SONARPY-2024
  y = x or 'str'
  y() # FN Value calculation with involved parameters support is needed (SONARPY-2023)
  z = x + 42
  z() # FN Value calculation with involved parameters support is needed (SONARPY-2023)

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
