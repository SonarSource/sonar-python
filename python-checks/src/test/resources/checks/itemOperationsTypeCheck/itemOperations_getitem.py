def builtins_supporting_getitem():
  mylist = ['a', 'b']
  mylist[0]

  mydict = {'a': 1, 'b': 2}
  mydict['a']

  bytearray(b"test")[1]

  # list and dict Comprehension
  [nb for nb in range(5)][0]
  {nb: 'a' for nb in range(4)}[0]

  range(10)[0]

  bytes(b'123')[0]
  memoryview(bytearray(b'abc'))[0]

  "abc"[0]
  (1, 2)[0]
  unknown_symbol[1] # Unknown symbol
  unknown_symbol()[1] # Unknown symbol

def builtins_not_supporting_getitem():
  # dictviews https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
  mydict = {'a': 1, 'b': 2}
  mydict.keys()[0] # Noncompliant
  mydict.values()[0] # Noncompliant
  mydict.items()[0] # Noncompliant

  # iterators
  iter(mylist)[0]  # Noncompliant

  # Numeric types
  from fractions import Fraction
  from decimal import Decimal
  1[0]  # Noncompliant
  1.0[0]  # Noncompliant
  complex(1,1)[0]  # Noncompliant {{Fix this code; this expression does not have a "__getitem__" method.}}
# ^^^^^^^^^^^^
  Fraction(1,1)[0]  # Noncompliant
  Decimal(1)[0]  # Noncompliant
  True[0]  # Noncompliant

  # Set
  {1}[0]  # Noncompliant
  # frozenset
  frozenset({1})[0]  # Noncompliant

  # set Comprehension
  {nb for nb in range(4)}[0]  # Noncompliant

  var = None
  var[0]  # Noncompliant {{Fix this code; "var" does not have a "__getitem__" method.}}
# ^^^^^^

  NotImplemented[0]  # FN: Any type

  def function(): ...
#     ^^^^^^^^> {{Definition of "function".}}

  function[0]  # Noncompliant
# ^^^^^^^^^^^

  def generator():
      yield 1

  generator()[0]  # FN: type unknown
  (nb for nb in range(5))[0]  # Noncompliant

  async def async_function(): ...

  async_function()[0]  # Noncompliant

  async def async_generator():
      yield 1

  async_generator()[0]  # Noncompliant
  open("foo.py")[0]  # Noncompliant


def standard_library():
  from array import array
  a = array('b', [0, 1, 2])
  a[0]

  from collections import namedtuple, deque, ChainMap, Counter, OrderedDict, defaultdict, UserDict, UserList, UserString

  Coord = namedtuple('Coord', ['x', 'y'])
  Coord(x=1, y=1)[0]

  deque([0,1,2])[0]
  ChainMap({'a': 1})['a']
  Counter(['a', 'b'])['a']
  OrderedDict.fromkeys('abc')['a']
  defaultdict(int, {0:0})[0]
  import math
  math[0]  # FN: type unknown

  import os
  os.popen('ls')[0]  # Noncompliant

def custom_classes():
  class A:
#       ^>
      def __init__(self, values):
          self._values = values

  a = A([0,1,2])

  a[0]  # Noncompliant
# ^^^^

  class B: ...

  B[0]  # Noncompliant


  class C:
      def __init__(self, values):
          self._values = values

      def __getitem__(self, key):
          return self._values[key]

  c = C([0,1,2])
  c[0]

  class D:
      def __class_getitem__(cls, key):
          return [0, 1, 2, 3][key]

  D[0]


def getitem(self, key):
    print(f"getting {key}")

def meta_classes():
  class MyMetaClassWithGet(type):
      def __new__(cls, name, bases, dct):
          instance = super().__new__(cls, name, bases, dct)
          instance.__getitem__ = getitem
          return instance

      def __getitem__(cls, key):
          print(f"getting {key}")

  class MetaclassedWithGet(metaclass=MyMetaClassWithGet): ...

  MetaclassedWithGet[0]  # OK
  MetaclassedWithGet()[0]  # OK


  class MyMetaClassWithoutGet(type): ...
  class MetaclassedWithoutGet(metaclass=MyMetaClassWithoutGet): ...

  MetaclassedWithoutGet[0]  # FN
  MetaclassedWithoutGet()[0]  # FN

def type_annotations():
  """No issue as type annotations do no call item methods"""
  from typing import Awaitable
  def my_func() -> Awaitable[bool]: ... # OK
  def my_other_func(arg: Awaitable[bool]): ... # OK
  x: Awaitable[bool] # OK
  Awaitable[None]
  from collections import Set
  CustomSet = Set[str]

def decorated_classes():
  import enum
  @enum.unique
  class MyEnum(enum.Enum):
      first = 0
      second = 1

  print(MyEnum["first"]) # OK

class A:
  def __init__(self):
    self.data = [1, 2, 3]
  def data(self):
    ...
  def f(self):
    self.data[1] # OK

def python3_9():
  from asyncio import Future
  class A(Future[TSource]): ...


def python3_10():
  type_alias = type[Exception]


def import_path():
    from importlib import import_module

    path = import_module('importlib').__path__[0]  # OK ref: SONARPY-1339

from ctypes import cast

def ctypes_cast(buf, sal):
  addrList = cast(buf, POINTER(sal))
  addrCount = addrList[0].iAddressCount # FN ref: SONARPY-1477

# We should not raise any issues on mocks as they could be monkey patched to be anything
def mocks():
    from unittest.mock import Mock
    mock = Mock()
    a = mock[42]

    class ExtendedMock(Mock):
      ...

    def custom_mock():
        a = ExtendedMock()[42]


class MyGenericClass[T]: ...

class MyGenericSubType(MyGenericClass[str]): ...
