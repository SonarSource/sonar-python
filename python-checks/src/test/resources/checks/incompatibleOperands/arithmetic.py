################################################
# Case to cover: Detect whem custom types do not
# have the right methods.
################################################

def custom_noncompliant():
  class Empty:
    pass

  class LeftAdd:
    def __add__(self, other):
      return 42

  class RightAdd:
    def __radd__(self, other):
      return 21

  class AddAssign:
    def __iadd__(self, other):
      return 42

  Empty() + 1  # Noncompliant
  LeftAdd() + 1  # Ok
  1 + LeftAdd()  # Noncompliant
  Empty() + LeftAdd()  # Noncompliant
  LeftAdd() + Empty()  # Ok

  RightAdd() + 1  # Noncompliant
  1 + RightAdd()  # Ok
  Empty() + RightAdd()  # Ok
  RightAdd() + Empty()  # Noncompliant


  empty = Empty()
  add_assign = AddAssign()
  empty += 1
  add_assign += 1

  -Empty()  # Noncompliant


def custom_compliant():
  class A:
    def __add__(self, other):
      return 42

    def __neg__(self):
      return -1

  a = A()
  myvar = A() + 1
  myvar = -a


#################################################
# Case to cover: Detect whem builtin types do not
# have the right methods.
#################################################

def builtin_noncompliant():
  1 + "1"  # Noncompliant
  "1" + 1  # Noncompliant
  1 + [1]  # Noncompliant
  1 + {1}  # Noncompliant
  1 + (1,)  # Noncompliant
  1 + {'a': 1}  # Noncompliant
  [1] + (1,)  # Noncompliant
  "foo " + "bar".encode('base64') # OK, FP in Python2
  "bar".encode('base64') + "foo"  # OK, FP in Python2


  -'1'  # Noncompliant

  from queue import Queue
  q = Queue()
  q.maxsize + 'other' # Noncompliant


def builtin_compliant():
  1 + int("1")
  myvar = -int('1')
  myvar += 2

  not myvar # coverage


def type_annotations():
  mode: "OpenBinaryMode" | "OpenTextMode"

#################################################
# Mocks could be monkey patched to possess any special methods
# No issues should be raised on them
#################################################
def mocks():
  from unittest.mock import Mock, MagicMock 
  mock = Mock()
  myvar = Mock() + 1
  myvar = -mock
  mock = MagicMock()
  mock += 1


  class ExtendedMock(MagicMock):
      ...

  def custom_mock():
    ExtendedMock() + 1
    -ExtendedMock()
