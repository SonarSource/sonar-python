def builtin_types_supporting_delitem():
  mylist = ['a', 'b']
  del mylist[0]

  mydict = {'a': 1, 'b': 2}
  del mydict['a']

  del bytearray(b"test")[1]

  # list and dict Comprehension
  del [nb for nb in range(5)][0]
  del {nb: 'a' for nb in range(4)}[0]
  del something(mydict[a]) # coverage (syntax error, not a __delitem__)


def builtin_types_not_supporting_delitem():
  # dictviews https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
  mydict = {'a': 1, 'b': 2}
  del mydict.keys()[0]  # Noncompliant
  del mydict.values()[0] # Noncompliant
  del mydict.items()[0]  # Noncompliant

  # iterators
  del iter(mylist)[0]  # Noncompliant

  # Numeric types
  from fractions import Fraction
  from decimal import Decimal
  del 1[0]  # Noncompliant
  del 1.0[0]  # Noncompliant
  del complex(1,1)[0]  # Noncompliant {{Fix this code; this expression does not have a "__delitem__" method.}}
#     ^^^^^^^^^^^^
  del Fraction(1,1)[0]  # Noncompliant
  del Decimal(1)[0]  # Noncompliant
  del True[0]  # Noncompliant {{Fix this code; "True" does not have a "__delitem__" method.}}
#     ^^^^^^^

  # set
  del {1}[0]  # Noncompliant
  # frozenset
  del frozenset({1})[0]  # Noncompliant

  # set Comprehension
  del {nb for nb in range(4)}[0]  # Noncompliant

  del range(10)[0]  # Noncompliant

  var = None
  del var[0]  # Noncompliant

  del bytes(b'123')[0]  # Noncompliant
  del memoryview(bytearray(b'abc'))[0]  # Noncompliant

  del "abc"[0]  # Noncompliant
  del (1, 2)[0]  # Noncompliant

  del NotImplemented[0]  # FN: Any type

  def function():
      pass

  del function[0]  # Noncompliant

  def generator():
      yield 1

  del generator()[0]  # FN: type unknown
  del (nb for nb in range(5))[0]  # Noncompliant

  async def async_function():
      pass

  del async_function()[0]  # Noncompliant

  async def async_generator():
      yield 1

  del async_generator()[0]  # Noncompliant

def standard_library():
  import math
  del math[0]  # FN: Symbol "OTHER" Any type

  # File
  del open("foo.py")[0]  # Noncompliant

  import os
  del os.popen('ls')[0]  # Noncompliant

  from array import array
  a = array('b', [0, 1, 2])
  del a[0]

  from collections import namedtuple, deque, ChainMap, Counter, OrderedDict, defaultdict, UserDict, UserList, UserString

  Coord = namedtuple('Coord', ['x', 'y'])
  del Coord(x=1, y=1)[0]  # FN: namedtuple type is unresolved

  del deque([0,1,2])[0]
  del ChainMap({'a': 1})['a']
  del Counter(['a', 'b'])['a']
  del OrderedDict.fromkeys('abc')['a']
  del defaultdict(int, {0:0})[0]


def custom_class():
  class A:
      def __init__(self, values):
          self._values = values

  a = A([0,1,2])

  del a[0]  # Noncompliant

  class B:
      pass

  del B[0]  # Noncompliant


  class C:
      def __init__(self, values):
          self._values = values

      def __delitem__(self, key):
          del self._values[key]

  c = C([0,1,2])

  del c[0]


def delitem(self, key):
    print(f"deleting {key}")

def meta_classes():
  class MyMetaClassWithDelete(type):
      def __new__(cls, name, bases, dct):
          instance = super().__new__(cls, name, bases, dct)
          instance.__delitem__ = delitem
          return instance

      def __delitem__(cls, key):
          print(f"deleting {key}")

  class MetaclassedWithDelete(metaclass=MyMetaClassWithDelete):
      pass

  del MetaclassedWithDelete[0]  # OK
  del MetaclassedWithDelete()[0]  # OK


  class MyMetaClassWithoutDelete(type):
      pass

  class MetaclassedWithoutDelete(metaclass=MyMetaClassWithoutDelete):
      pass

  del MetaclassedWithoutDelete[0]  # FN
  del MetaclassedWithoutDelete()[0]  # FN


def import_path():
    from importlib import import_module

    del import_module('importlib').__path__[0]  # OK ref: SONARPY-1339


# We should not raise any issues on mocks as they could be monkey patched to be anything
def mocks():
    from unittest.mock import Mock, MagicMock
    mock = Mock()
    del mock[42]
    mock = MagicMock()
    del mock[0]
