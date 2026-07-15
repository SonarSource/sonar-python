from typing import SupportsFloat, List, Iterable, Generator, Set, Union, Type, TypedDict
from inconsistentTypeHintImported import ClassWithFieldOnly, ClassWithMethodOnly

def assigned_directly():
  foo : int = None # Noncompliant {{Replace the type hint "int" with "Optional[int]" or don't assign "None" to "foo"}}
#       ^^^>  ^^^^
  my_int_nok: int = "hello"  # Noncompliant  {{Assign to "my_int_nok" a value of type "int" instead of "str" or update its type hint.}}
#             ^^^>  ^^^^^^^
  my_str_ok: str = 42  # Noncompliant
  my_int_ok: int = 42  # OK
  my_str_ok: str = "hello"  # OK
  a : ClassWithFieldOnly = None # Noncompliant
  b : ClassWithMethodOnly = None # Noncompliant

def return_union() -> Union[str, float]:
  ...

def assigned_to_union(cond):
  my_int_nok: int = return_union()  # Noncompliant  {{Assign to "my_int_nok" a value of type "int" instead of "Union[str, float]" or update its type hint.}}
  if cond:
    x = "hello"
  else:
    x = 42.5
  my_int_nok: int = x  # Noncompliant  {{Assign to "my_int_nok" a value of type "int" or update its type hint.}}


def assigned_later(param):
  a: int
  a = "hello"  # FN
  b: int
  if param:
    b = 42
  else:
    b = "42"  # FN

  c: int
  c = 1 if a else 2  # OK
  d: int
  d = 1 if a else "hello"  # FN


def custom_classes():
  class A:
    def method(): ...

  class B(A):
    def additional_method(): ...

  my_a_ok: A = A()  # OK
  my_a_ok2: A = B()  # OK
  my_a_nok: A = A  # Noncompliant
  my_b_nok: B = A()  # Noncompliant {{Assign to "my_b_nok" a value of type "B" instead of "A" or update its type hint.}}
#           ^>  ^^^
  my_b_ok: B = B()


def get_generator():
  yield 1


def type_aliases():
  """We should avoid raising FPs on type aliases"""
  my_float: SupportsFloat = 42  # OK
  my_iterable: Iterable = []  # OK
  my_generator: Generator = get_generator()  # OK


def collections():
  my_list: List = {}  # Noncompliant

  my_str_list_nok: List[str] = [1, 2, 3]  # FN

  my_str_list_ok: List[str] = ["a", "b", "c"]  # OK

  my_set_nok: Set = {}  # Noncompliant {{Assign to "my_set_nok" a value of type "set" instead of "dict" or update its type hint.}}

  my_set_nok2: Set = set  # Noncompliant  {{Assign to "my_set_nok2" a value of type "set" instead of "type" or update its type hint.}}

  my_set_ok: Set = set()  # OK


def function_params():
  def overwritten_param(param: int):
    param = "hello"  # Out of scope (S1226)

  def used_param(param: int):
    print(param)
    param = "hello"  # FN
    print(param)


class ClassAttributes:
  my_attr: str = "hello"  # OK
  my_attr_2: str = 42  # Noncompliant

  my_attr_3: str

  def __init__(self):
    self.my_attr_3 = 42  # FN
    self.my_attr_4: int = "hello"  # Noncompliant {{Assign to this expression a value of type "int" instead of "str" or update its type hint.}}

class Meta(type): ...

class MyClassWithMeta(metaclass=Meta): ...

def metaclasses():
  my_var: Meta = set # Accepted FN
  my_other_var: Meta = MyClassWithMeta  # OK
  my_other_var: MyClassWithMeta = MyClassWithMeta  # Noncompliant {{Assign to "my_other_var" a value of type "MyClassWithMeta" instead of "type" or update its type hint.}}
  another_var: Type = MyClassWithMeta
  another_var: Type = set
  def a_function(): ...
  another_var: Type = a_function  # Accepted FN
  another_var: Type = unknown_symbol

def type_dict():
  class MyCustomDict(TypedDict):
    user_ids: Set[int]
    message_ids: Set[int]

  def my_dict() -> MyCustomDict:
    users = {1,2,3}
    messages = {1,2,3}
    my_dict: MyCustomDict = dict(user_ids=users, message_ids=messages)  # OK
    return my_dict
