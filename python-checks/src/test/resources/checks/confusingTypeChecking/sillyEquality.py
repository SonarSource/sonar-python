class A: ...
class B: ...
class ComparableA:
  def __eq__(self, other): ...
class ComparableB:
  def __ne__(self, other): ...

def custom(x: A, y: B):
  if x == y: ... # Noncompliant {{Fix this equality check; Previous type checks suggest that operands have incompatible types.}}

def custom_comparable(x: A, y: ComparableA, z: ComparableB):
  if x == y: ...
  if x == x: ...
  if x == z: ...

def builtins(x: int, y: str):
  if x == y: ... # Noncompliant
  if x == 'foo': ... # Noncompliant
  if 42 == y: ... # Noncompliant
  if x == None: ... # OK
  if None == y: ... # OK

type T = str
def foo(a: T):
  if a == 42: # FN
    ...


from unittest.mock import Mock, MagicMock
# We should not raise any issues on mocks as they could be monkey patched to fit the comparison type
def mocks(mock: Mock, magic_mock:MagicMock):
    mock == 3 # Ok
    mock != 42
    3 == magic_mock # Ok
