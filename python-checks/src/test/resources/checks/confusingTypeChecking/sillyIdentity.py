from typing import Union
class A:
  def foo(self): ...
class B: ...
def custom(x: A, y: B, z: Union['foo', 'bar']):
  if x is y: ... # Noncompliant {{Fix this identity check; Previous type checks suggest that operands have incompatible types.}}
  #    ^^
  if x is not y: ... # Noncompliant
  #    ^^^^^^
  if x is x: ...
  xx = x
  if x:
    yy = xx
  else:
    yy = y
  if yy is not xx: ...
  if z is x: ...
  if x is z: ...

def builtins(x: int, y: str):
  if x is y: ... # Noncompliant
  if x is 'foo': ... # Noncompliant
  if 42 is y: ... # Noncompliant
  if x is None: ... # OK
  if None is y: ... # OK

class Base: ...
class C(Base): ...
def with_superclass(x: Base, y: C):
  if x is y: ... # OK
  if y is x: ... # OK
