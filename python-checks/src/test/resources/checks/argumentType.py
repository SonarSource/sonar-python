from math import acos
import datetime
import time
import select
import genericpath
import _heapq
import imaplib
from unittest.mock import Mock
from typing import Dict, Tuple, Set
from collections import OrderedDict, Counter
from emoji import emojize

class ExpectedClass():
  a = 42
  def expected_method(): ...
class ExpectedSubClass(ExpectedClass): ...
class DuckTypeCompatibleClass:
  a = 42
  def expected_method(): ...
  def some_other_method(): ...
class UnexpectedClass(): a = 42

def functions_defined_locally():
  def function_with_int(a: int): ...
  function_with_int("154") # Noncompliant
  function_with_int(154)  # OK

  def function_with_custom_type_arg(smth: ExpectedClass, a: int): ...
  unexpected = UnexpectedClass()
  expected = ExpectedClass()
  my_int = 42
  ducktyped = DuckTypeCompatibleClass()
  function_with_custom_type_arg(unexpected) # Noncompliant
  function_with_custom_type_arg(a = my_int, smth = expected)
  function_with_custom_type_arg(a = my_int, smth = unexpected) # Noncompliant
  function_with_custom_type_arg(expected, my_int, 42) # S930 will handle this
  function_with_custom_type_arg(ducktyped, my_int) # OK, a class is ducktype compatible with another if it has the same members and methods

  def function_with_keyword_only(smth: ExpectedClass, *, other: ExpectedClass): ...
  function_with_keyword_only(expected, other = expected)
  function_with_keyword_only(expected, other = unexpected) # Noncompliant {{Change this argument; Function "function_with_keyword_only" expects a different type}}
#                                      ^^^^^^^^^^^^^^^^^^

  def function_with_positional_only(smth: ExpectedClass, /, other: ExpectedClass): ...
  function_with_positional_only(expected, expected)
  function_with_positional_only(expected, other = expected)
  function_with_positional_only(expected, other = unexpected) # Noncompliant
  function_with_positional_only(expected, unexpected) # Noncompliant

  class SomeClass():
    def method_with_positional_and_keyword_only(self, smth: ExpectedClass, /, other: int, *, then: ExpectedClass): ...
  my_SomeClass = SomeClass()
  my_SomeClass.method_with_positional_and_keyword_only(expected, 42, then = ExpectedSubClass())
  my_SomeClass.method_with_positional_and_keyword_only(expected, my_int, then = ExpectedSubClass())
  my_SomeClass.method_with_positional_and_keyword_only(expected, 42, then = UnexpectedClass()) # Noncompliant
  my_SomeClass.method_with_positional_and_keyword_only(expected, 42, then = unexpected) # Noncompliant

def stdlib_functions():
  A = UnexpectedClass()
  acos(A) # Noncompliant
  B = datetime.tzinfo()
  B.tzname(42) # Noncompliant {{Change this argument; Function "tzname" expects a different type}}
#          ^^
  select.select([],[],[], 0)
  time.sleep(1) # OK
  x = time.gmtime(int(time.time()))
  x = time.gmtime(secs = int(time.time()))
  time.sleep(True) # OK, converted to 1
  time.sleep(1j) # FN, considered duck type compatible
  genericpath.isfile("some/path")
  genericpath.isfile(42) # Noncompliant
  my_list = [1,2,3]
  _heapq.heapify(42) # Noncompliant {{Change this argument; Function "heapify" expects a different type}}
#                ^^
  _heapq.heapify(my_list)
  imap4 = imaplib.IMAP4()
  imap4.setannotation(42) # FN, we do not handle variadic parameters
  imap4.setannotation("some string") # OK
  str_tuple = "string", "another string"
  imap4.setannotation(str_tuple) # OK

def builtin_functions():
  round(42.3)
  round("42.3")  # FN, ambiguous symbol: no parameters defined yet | missing type hierarchy
  unexpected = UnexpectedClass()
  number = 42
  number.__add__("27") # Noncompliant
  number.__add__(unexpected) # Noncompliant
  number.__add__(x = unexpected) # Noncompliant {{Change this argument; Function "__add__" expects a different type}}
#                ^^^^^^^^^^^^^^
  float.fromhex(42) # Noncompliant
  eval(42) # Noncompliant
  "Some string literal".format(1, 2)
  exit(1)
  repr(A)
  arr = []
  len(arr) # OK, duck type compatibility
  values = OrderedDict((key, 0) for key in field_order)
  len(values)
  tld_counter = Counter()
  len(tld_counter)

  str.ljust(str(1), 3)

  x = {}
  x = []
  x.pop(()) # Noncompliant

def third_party_functions():
  emojize("Python is :thumbs_up:") # OK
  emojize(42) # Noncompliant

def type_aliases():
  def with_set(a : Set[int]): ...
  def with_dict(a : Dict[int, int]): ...
  def with_tuple(a : Tuple[int]): ...

  with_set({42}) # OK
  with_set({1 : 42}) # Noncompliant
  with_set((42, 43)) # Noncompliant
  with_set(a = 42) # Noncompliant

  with_dict({42}) # Noncompliant
  with_dict({1 : 42}) # OK
  with_dict((42, 43)) # Noncompliant
  with_dict(42) # Noncompliant

  with_tuple({42}) # Noncompliant
  with_tuple({1 : 42}) # Noncompliant
  with_tuple((42, 43)) # OK
  with_tuple(42) # Noncompliant

def edge_cases():
  ambiguous = 42
  def ambiguous(a: str): ...
  ambiguous(42) # OK
  def func(a: int, b: int): ...
  func(b = 42, 42) # not a valid syntax
  func(*unpack) # OK
  unknown_call(1,2,3)
  def func_no_parameters(): ...
  func_no_parameters()
  def func_only_keywords(*, arg: str): ...
  func_only_keywords(arg = 42) # Noncompliant

  class SomeClass():
    def my_method(self, x: int): ...
    @staticmethod
    def static_method(y: str): ...
    @classmethod
    def class_method(cls, y: str): ...
    def ambiguous_static_method(y: str): ...
    @unknowndecorator
    def method_with_unknowndecorator(y: str): ...
    @decorator1
    @decorator2
    def method_with_multiple_decorators(y: str): ...
  A = SomeClass()
  A.my_method("42") # Noncompliant
  A.my_method(42)
  SomeClass.ambiguous_static_method("some string")
  SomeClass.ambiguous_static_method(42) # Noncompliant
  A.static_method(42) # Noncompliant
  SomeClass.static_method(42) # Noncompliant
  A.class_method(42) # Noncompliant
  SomeClass.class_method(42) # Noncompliant
  A.method_with_unknowndecorator(42) # OK, unknown decorator
  A.method_with_multiple_decorators(42) # OK, multiple decorators

def exception_for_unittest_mock():
  class SomeMock(Mock): ...
  my_mock = SomeMock()
  hex(my_mock) # OK

def exception_for_object():
  def foo(a: object): ...
  foo([])

class StaticCallInsideClass:
  def my_method(a: int, b: str): ...
  my_method(1, "hello") # OK

def not_static_call():
  class MyClass:
    def foo(self, x: int, y: str): ...
  a = MyClass()
  f = a.foo
  f("hello", "hello") # FN
  f(42, "hello") # OK


def duck_typing_no_member():
  class Parent():
    def do_something(self): ...


  class ChildA(Parent): ...

  class ChildB(Parent):
    def do_something_else(): ...

  def a_function(param: ChildA):
    param.do_something()

  def another_function(param: ChildB):
    param.do_something_else()

  a_function(Parent())  # OK, still duck type compatible with ChildA
  another_function(Parent())  # Noncompliant
