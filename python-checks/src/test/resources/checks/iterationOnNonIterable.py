def every_kind_of_iteration():
  iterable = ['a', 'b', 'c']
  not_an_iterable = 42
  # unpacking arguments
  print(*iterable)
  print(*not_an_iterable)  # Noncompliant {{Replace this expression with an iterable object.}}
#        ^^^^^^^^^^^^^^^

  # for-in loop
  for a in iterable: ...

  for a in not_an_iterable: ... # Noncompliant {{Replace this expression with an iterable object.}}
#          ^^^^^^^^^^^^^^^

  for a in 1, 2: ... # OK

  for a in unknown_function(): ... # OK

  for a in unknown_symbol: ... # OK

  # unpacking
  a, *rest = iterable
  a, *rest = not_an_iterable # Noncompliant
  [a, *rest] = not_an_iterable # Noncompliant
  (a, *rest) = not_an_iterable # Noncompliant

  # comprehensions
  a, *rest = [a for a in iterable]
  a, *rest = {a for a in iterable}
  a, *rest = {a: a for a in iterable}


  a, *rest = [a for a in not_an_iterable]  # Noncompliant
  a, *rest = {a for a in not_an_iterable}  # Noncompliant
  a, *rest = {a: a for a in not_an_iterable}  # Noncompliant

  # yield from
  def yield_from():
    iterable = ['a', 'b', 'c']
    not_an_iterable = 42
    yield from iterable
    yield from not_an_iterable  # Noncompliant

def iteration_on_builtins():
  iterable = ['a', 'b', 'c']
  mydict = {"a": 1, "b": 2}

  # unpacking
  a, *rest = iterable
  a, *rest = iter(iterable)
  a, *rest = set(iterable)
  a, *rest = frozenset(iterable)
  a, *rest = iterable
  a, *rest = "abc"
  a, *rest = f"abc"
  a, *rest = u"abc"
  a, *rest = b"abc"
  a, *rest = bytes(b"abc")
  a, *rest = bytearray(b"abc")
  a, *rest = memoryview(b"abc")
  a, *rest = mydict.keys()
  a, *rest = mydict.values()
  a, *rest = mydict.items()
  a, *rest = range(10)

  # Numeric types
  from fractions import Fraction
  from decimal import Decimal
  a, *rest = 1  # Noncompliant
  a, *rest = 1.0  # Noncompliant
  a, *rest = complex(1,1)  # Noncompliant
  a, *rest = Fraction(1,1)  # Noncompliant
  a, *rest = Decimal(1)  # Noncompliant
  a, *rest = True  # Noncompliant
  a, *rest = None  # Noncompliant
  a, *rest = NotImplemented  # FN: Any type

  def function(): ...
#     ^^^^^^^^> {{Definition of "function".}}
  a, *rest = function  # Noncompliant
#            ^^^^^^^^

  # generators
  def generator():
    yield 1

  a, *rest = generator()
  a, *rest = generator  # Noncompliant

def standard_library():
  import csv
  reader = csv.reader(handle, delimiter=",", quotechar='"')
  for line in reader: ... # OK
  for x in csv.reader(f): ... # OK

def dict_unpacking():
  not_a_dict = 42
  dict(**not_a_dict)  # Out of scope

def custom_types():
  class NewStyleIterable:
    li = [1,2,3]

    def __iter__(self):
      return iter(self.__class__.li)

  class OldStyleIterable:
    li = [1,2,3]

    def __getitem__(self, key):
      return self.__class__.li[key]


  a, *rest = NewStyleIterable()
  a, *rest = OldStyleIterable()
  a, *rest = NewStyleIterable  # Noncompliant
  a, *rest = OldStyleIterable  # Noncompliant

  class Empty(): ...

  a, *rest = Empty()  # Noncompliant

  class NonIterableClass:
    li = [1,2,3]

    def __class_getitem__(cls, key):
      "__class_getitem__ does not make a class iterable"
      return cls.li[key]

  a, *rest = NonIterableClass  # Noncompliant

  # Inheritance
  class customTuple(tuple): ...
  a, *rest = customTuple([1,2,3])

def async_iteration():
  async def async_function(): ...

  a, *rest = async_function()  # Noncompliant
  a, *rest = async_function  # Noncompliant

  async def async_generator():
    yield 1

  a, *rest = async_generator()  # Noncompliant
  for a in async_generator():  # Noncompliant {{Add "async" before "for"; This expression is an async generator.}}
#          ^^^^^^^^^^^^^^^^^
    print(a)

  async for a in async_generator(): ...
  async for a in unknown(): ... # OK

  class AsyncIterable:
    def __aiter__(self):
      return AsyncIterator()

  class AsyncIterator:
    def __init__(self):
      self.start = True

    async def __anext__(self):
      if self.start:
        self.start = False
        return 42
      raise StopAsyncIteration

  async for a in AsyncIterable(): ...

  async for a in 42: ... # FN (out of scope)

  for a in AsyncIterable(): ... # Noncompliant

def metaclasses():
  """Out of scope: can't be sure the metaclass doesn't add the required method"""
  class MyMetaClassWithoutIter(type): ...

  class MetaclassedNonIterable(metaclass=MyMetaClassWithoutIter): ...

  # Accepted FNs
  a, *rest = MetaclassedNonIterable  # FN
  a, *rest = MetaclassedNonIterable() # FN

  def myiter(self):
    return iter(range(10))

  class MyMetaClassWithIter(type):
    def __new__(cls, name, bases, dct):
      instance = super().__new__(cls, name, bases, dct)
      instance.__iter__ = myiter
      return instance

  class MetaclassedIterable(metaclass=MyMetaClassWithIter): ...

  a, *rest = MetaclassedIterable() # OK

def attributes_and_properties():
  """Out of scope: Detecting when a non-iterable class and instance attribute is iterated over."""
  class MyContainer:
    def __init__(self):
        self._mylist = None

    @property
    def mylist(self):
      if not self._mylist:
        self._mylist = [1, 2, 3]
      return self._mylist

  a, *rest = MyContainer().mylist

  class AbstractClass():
    attribute_set_in_subclass = None

    def process(self):
      for a in self.attribute_set_in_subclass:  # FN (out of scope)
        print(a)


def calling_iter_with_non_iterable():
  not_an_iterable = 42
  iter(not_an_iterable)  # FN (out of scope)

def arrays_no_issue():
  from array import array
  a, *rest = array('b', [0, 1, 2])

def collections_no_issue():
  from collections import namedtuple, deque, ChainMap, Counter, OrderedDict, defaultdict, UserDict, UserList, UserString

  Coord = namedtuple('Coord', ['x', 'y'])
  a, *rest = Coord(x=1, y=1)

  a, *rest = deque([0,1,2])
  a, *rest = ChainMap({'a': 1})
  a, *rest = Counter(['a', 'b'])
  a, *rest = OrderedDict.fromkeys('abc')
  a, *rest = defaultdict(int, {0:0})

def inherits_from_metaclassed():
  import enum
  class MyEnum(enum.Enum):
      first = 0
      second = 1

  for elem in MyEnum: ... # OK

def enum_unresolved_type_hierarchy():
  try:
    from enum import Enum
  except ImportError:
    from foo.enum import Enum

  class MyEnumFoo(Enum):
    first = 0
    second = 1

  for elem in MyEnumFoo: ... # OK


# We should not raise any issues on mocks as they could be monkey patched to fit any type
def mocks_no_issue():
  from unittest.mock import Mock, MagicMock

  mock = Mock()
  a, *rest = mock
  iter(mock)
  mock = MagicMock()
  for elem in mock: ... # OK

 

  class ExtendedMock(MagicMock):
    ...

  def custom_mock():
    extended_mock = ExtendedMock()
    a, *rest = extended_mock
    iter(extended_mock)
    for elem in extended_mock: ... # OK
