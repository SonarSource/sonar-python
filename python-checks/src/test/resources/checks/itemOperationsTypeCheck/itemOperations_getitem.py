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

  @some_decorator
  class ClassWithDecorator: ...
  ClassWithDecorator[0]  # FN: decorator


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


def lxml_fromstring_returns_element():
  from lxml import etree

  root = etree.fromstring(b"<response><status>ok</status></response>")
  root[0]  # The custom stub must expose _Element, not _ElementTree.


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


def generic_cases(unknown_type):
  from typing import Annotated, Literal
  from importedGeneric import ImportedGeneric

  class MyGenericClass[T]: ...

  class TwoParameterGeneric[T, U]: ...

  class ParamSpecGeneric[**P, R]: ...

  class TypeVarTupleGeneric[*Ts, R]: ...

  class MyGenericSubType(MyGenericClass[str]): ...

  class SomeOtherClass: ...
  SomeOtherClassAlias = SomeOtherClass

  class Namespace:
    class Type[T]: ...

  T = TypeVar('T')
  GenericAlias = MyGenericClass[T]  # OK
  IntAlias = MyGenericClass[int]  # OK
  IntLiteralAlias = MyGenericClass[0]  # Noncompliant
  StrAlias = MyGenericClass[str]  # OK
  StrLiteralAlias = MyGenericClass["str"]  # OK
  ParenthesizedAlias = MyGenericClass[(SomeOtherClass)]  # OK
  UnionAlias = MyGenericClass[SomeOtherClass | MyGenericSubType]  # OK
  UnionWithNoneAlias = MyGenericClass[SomeOtherClass | None]  # OK
  InvalidUnionRightAlias = MyGenericClass[SomeOtherClass | 0]  # Noncompliant
  InvalidUnionLeftAlias = MyGenericClass[0 | SomeOtherClass]  # Noncompliant
  InvalidBinaryAlias = MyGenericClass[SomeOtherClass + 0]  # Noncompliant
  QualifiedAlias = MyGenericClass[Namespace.Type]  # OK
  ParameterizedQualifiedAlias = MyGenericClass[Namespace.Type[SomeOtherClass]]  # OK
  ImportedAlias = ImportedGeneric[SomeOtherClass]  # OK
  LiteralAlias = MyGenericClass[Literal[0]]  # OK
  AnnotatedAlias = MyGenericClass[Annotated[SomeOtherClass, "metadata"]]  # OK
  InvalidTwoParameterAlias = TwoParameterGeneric[int, 0]  # Noncompliant
  NestedUnionAlias = MyGenericClass[MyGenericClass[SomeOtherClass] | MyGenericClass[MyGenericSubType]]  # OK
  EllipsisAlias = ParamSpecGeneric[..., int]  # OK
  ParamSpecAlias = ParamSpecGeneric[[int, str], int]  # OK
  TypeVarTupleAlias = TypeVarTupleGeneric[*Ts, int]  # OK
  ParameterizedEllipsisAlias = TwoParameterGeneric[str, tuple[int, ...]]  # OK
  TupleAlias = TwoParameterGeneric[(int, str)]  # OK
  InvalidTupleAlias = TwoParameterGeneric[(int, 0)]  # Noncompliant

  from typing import Annotated, Callable, Literal, TypeVar, TypeVarTuple
  from typing import Literal as Lit
  from typing_extensions import Annotated as ExtAnnotated, Literal as ExtLiteral
  import typing

  TwoParameterGeneric[Literal[1], str]  # OK
  TwoParameterGeneric[Literal[-1, 0, 1], str]  # OK
  TwoParameterGeneric[Annotated[int, 42], str]  # OK
  TwoParameterGeneric[Annotated[int, object()], str]  # OK
  TwoParameterGeneric[list[Literal[1]], str]  # OK
  TwoParameterGeneric[Annotated[Literal[1], 42], str]  # OK
  TwoParameterGeneric[Lit[1], str]  # OK
  TwoParameterGeneric[typing.Literal[1], str]  # OK
  TwoParameterGeneric[ExtLiteral[1], str]  # OK
  TwoParameterGeneric[ExtAnnotated[int, 42], str]  # OK

  NestedT = TypeVar("NestedT")
  class Nested(MyGenericClass[list[NestedT]]): ...
  class Deep(MyGenericClass[dict[str, list[NestedT]]]): ...
  class Callback(MyGenericClass[Callable[[NestedT], int]]): ...
  class Fixed(MyGenericClass[list[int]]): ...
  class MetadataOnly(MyGenericClass[Annotated[int, NestedT]]): ...
  NestedTs = TypeVarTuple("NestedTs")
  class Variadic(TypeVarTupleGeneric[*NestedTs, int]): ...

  Nested[int]  # OK
  Deep[int]  # OK
  Callback[int]  # OK
  Fixed[str]  # Noncompliant
  MetadataOnly[str]  # Noncompliant
  Variadic[str]  # OK

  from typing import ParamSpec
  from typing_extensions import TypeVar as ExtTypeVar
  from importedGeneric import ExportedT
  import importedGeneric

  CallbackP = ParamSpec("CallbackP")
  class ParamSpecChild(MyGenericClass[Callable[CallbackP, int]]): ...
  ParamSpecChild[int]  # OK

  ExtensionT = ExtTypeVar("ExtensionT")
  class ExtensionChild(MyGenericClass[ExtensionT]): ...
  ExtensionChild[int]  # OK

  class ImportedTypeVarChild(MyGenericClass[ExportedT]): ...
  class QualifiedTypeVarChild(MyGenericClass[importedGeneric.ExportedT]): ...
  ImportedTypeVarChild[int]  # OK
  QualifiedTypeVarChild[int]  # OK

  RenamedT = NestedT
  NestedAlias = list[NestedT]
  class RenamedTypeVarChild(MyGenericClass[RenamedT]): ...
  class TypeAliasChild(MyGenericClass[NestedAlias]): ...
  RenamedTypeVarChild[int]  # OK
  TypeAliasChild[int]  # OK

  a = MyGenericClass[int]()
  a = MyGenericClass[SomeOtherClass]()
  a = MyGenericClass[SomeOtherClassAlias]()
  b = MyGenericClass[unknown_type]()
  b = MyGenericClass[0]() # Noncompliant
  b = MyGenericClass["str"]() # OK

  c = SomeOtherClass()

  SomeOtherClass[SomeOtherClass | MyGenericSubType]  # Noncompliant
  c[0] # Noncompliant
  c[int] # Noncompliant
  c["int"] # Noncompliant

  class Loader:
    def __getitem__(self, item):
      return self.loader[item] # Noncompliant
    
    def loader(self):
      pass
