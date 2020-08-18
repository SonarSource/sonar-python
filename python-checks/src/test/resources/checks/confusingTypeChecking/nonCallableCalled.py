from typing import Set, FrozenSet, Union

def empty_union(x: Union['A', 'B']):
  x()

def foo(param1: int, param2: Set[int], param3: FrozenSet[int]):
  param1() # Noncompliant {{Fix this call; Previous type checks suggest that "param1" has type int and it is not callable.}}
  x = 42
  x() # OK, raised by S5756

  param2() # Noncompliant
  s = set()
  s() # OK, raised by S5756

  param3() # Noncompliant
  fs = frozenset()
  fs() # OK, raised by S5756

def derived(x: int):
  x.conjugate()() # NonCompliant
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
