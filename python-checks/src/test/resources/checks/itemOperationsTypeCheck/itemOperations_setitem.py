def builtin_types_supporting_setitem():
  mylist = ['a', 'b']
  mylist[0] = 42

  mydict = {'a': 1, 'b': 2}
  mydict['a'] = 42

  bytearray(b"test")[1] = 42

  [nb for nb in range(5)][0] = 42
  {nb: 'a' for nb in range(4)}[0] = 42

  # No issue raised on memoryview even when the memory is read-only
  memoryview(bytearray(b'abc'))[0] = 42
  memoryview(bytes(b'abc'))[0] = 42  # This will fail because bytes is read-only but we don't raise any issue
  a['b'].something = 42 # Not a __setitem__


def builtin_types_not_supporting_setitem():
  mydict = {'a': 1, 'b': 2}
  # dictviews https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
  mydict.keys()[0] = 42 # Noncompliant
  mydict.values()[0] = 42 # Noncompliant
  mydict.items()[0] = 42  # Noncompliant

  # iterators
  iter(mylist)[0] = 42  # Noncompliant

  # Numeric types
  from fractions import Fraction
  from decimal import Decimal
  1[0] = 42  # Noncompliant
  1.0[0] = 42  # Noncompliant
  complex(1,1)[0] = 42  # Noncompliant {{Fix this code; this expression does not have a "__setitem__" method.}}
# ^^^^^^^^^^^^
  Fraction(1,1)[0] = 42  # Noncompliant
  Decimal(1)[0] = 42  # Noncompliant
  True[0] = 42  # Noncompliant

  # Set
  {1}[0] = 42  # Noncompliant
  # frozenset
  frozenset({1})[0] = 42  # Noncompliant

  # set Comprehension
  {nb for nb in range(4)}[0] = 42 # Noncompliant

  range(10)[0] = 42  # Noncompliant

  var = None
  var[0] = 42  # Noncompliant

  bytes(b'123')[0] = 42  # Noncompliant

  # String
  "abc"[0] = 42  # Noncompliant
  # Tuple
  (1, 2)[0] = 42 # Noncompliant

  NotImplemented[0] = 42  # FN: Any type

  def function(): ...

  function[0] = 42  # Noncompliant

  def generator():
      yield 1

  generator()[0] = 42  # FN: unknown type
  (nb for nb in range(5))[0] = 42  # Noncompliant

  async def async_function(): ...

  async_function()[0] = 42  # Noncompliant

  async def async_generator():
      yield 1

  async_generator()[0] = 42  # Noncompliant

def standard_library():
  # module
  import math
  math[0] = 42  # FN: Any type

  # File
  open("foo.py")[0] = 42  # Noncompliant

  import os
  os.popen('ls')[0] = 42  # Noncompliant

  from array import array
  a = array('b', [0, 1, 2])
  a[0] = 42

  from collections import namedtuple, deque, ChainMap, Counter, OrderedDict, defaultdict, UserDict, UserList, UserString

  Coord = namedtuple('Coord', ['x', 'y'])
  Coord(x=1, y=1)[0] = 42 # FN: namedtuple type is unresolved

  deque([0,1,2])[0] = 42
  ChainMap({'a': 1})['a'] = 42
  Counter(['a', 'b'])['a'] = 42
  OrderedDict.fromkeys('abc')['a'] = 42
  defaultdict(int, {0:0})[0] = 42


def custom_classes():
  class A:
      def __init__(self, values):
          self._values = values

  a = A([0,1,2])

  a[0] = 42  # Noncompliant {{Fix this code; "a" does not have a "__setitem__" method.}}
# ^^^^

  class B: ...

  B[0]  # Noncompliant

  class C:
      def __init__(self, values):
          self._values = values

      def __setitem__(self, key, value):
          self._values[key] = value

  c = C([0,1,2])

  c[0] = 42


def setitem(self, key, value):
    print(f"setting {key}")

def meta_classes():
  class MyMetaClassWithSet(type):
      def __new__(cls, name, bases, dct):
          instance = super().__new__(cls, name, bases, dct)
          instance.__setitem__ = setitem
          return instance

      def __setitem__(cls, key, value):
          print(f"setting {key}")

  class MetaclassedWithSet(metaclass=MyMetaClassWithSet):
      pass

  MetaclassedWithSet[0] = 42  # OK
  MetaclassedWithSet()[0] = 42  # OK


  class MyMetaClassWithoutSet(type): ...

  class MetaclassedWithoutSet(metaclass=MyMetaClassWithoutSet): ...

  MetaclassedWithoutSet[0] = 42  # FN
  MetaclassedWithoutSet()[0] = 42  # FN


def import_path():
    from importlib import import_module

    import_module('importlib').__path__[0] = "test"  # OK ref: SONARPY-1339


# We should not raise any issues on mocks as they could be monkey patched to be anything
def mocks():
    from unittest.mock import Mock, MagicMock
    mock = Mock()
    mock[42] = 42
    mock = MagicMock()
    mock[0] = "foo"


    class ExtendedMock(MagicMock):
        ...

    def custom_mock():
        extended_mock = ExtendedMock()
        extended_mock[42] = 42
