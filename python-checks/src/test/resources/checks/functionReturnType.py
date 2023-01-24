from typing import List, SupportsFloat, Set, Dict, NoReturn, Text, Generator, Tuple, Union, AnyStr, Iterator, Iterable, Callable, Optional, TypedDict, AsyncIterator, AsyncGenerator
import numpy as np

def builtins():
  def my_str_nok() -> str:
    return 42  # Noncompliant

  def my_str_ok() -> str:
    return "hello"

  def my_int_nok() -> int:
  #   ^^^^^^^^^^>     ^^^>
    return "42"  # Noncompliant  {{Return a value of type "int" instead of "str" or update function "my_int_nok" type hint.}}
  #        ^^^^

  def my_int_ok() -> int:
    return 42

  def my_none_nok() -> None:
    return 42  # Noncompliant

  def my_none_ok() -> None:
    print("hello")

  def my_none_ok_2(param) -> None:
    if param:
      return
    print("hello")

  def my_object() -> object:
    d = dict()
    return d  # OK

def numbers():
  def my_float() -> float:
    return 0  # OK

  def my_optional_float() -> Optional[float]:
    return 0  # OK

def tuples():
  def my_tuple_nok() -> int:
    return 1, 2, 3  # Noncompliant

  def my_tuple_ok() -> tuple:
    return 1, 2, 3

  def my_tuple_ok_2(param) -> tuple:
    if param:
      x = 1, 2, 3
      return x
    else:
      import collections
      Person = collections.namedtuple('Person', ['name', 'age', 'gender'])
      return Person(name="Bob", age=30, gender="male")  # OK

  def my_tuple_ok_3(param) -> Union[tuple, bool]:
    if param:
      return 1, 2, 3
    else:
      return False

  def my_tuple_ok_4(param) -> Union[Tuple["a", "b"], bool]:
    if param:
      return 1, 2
    else:
      return False

  def my_union_unknown_type_tuple() -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    return 1, 2, 3  # OK

  def my_optional_tuple() -> Optional[Tuple]:
    return 1, 2, 3 # OK

  def my_one_element_tuple() -> tuple:
    return 1,  # OK

def collections():
  def my_list_nok() -> List:
    return 42  # Noncompliant

  def my_list_nok_2() -> List[int]:
    return 42  # Noncompliant

  def my_list_ok() -> List:
    return [42]

  def my_set_nok() -> Set:
    # {} is Empty dict literal
    return {}  # Noncompliant

  def my_set_nok_2() -> Set[str]:
    # {} is Empty dict literal
    return {}  # Noncompliant

  def my_set_ok() -> Set:
    return {42}  # OK

  def my_dict() -> Dict:
    return {}  # OK


def type_aliases():
  """We should avoid FPs for type aliases used as type hint"""
  def my_supports_float() -> SupportsFloat:
    return 42  # OK

  def my_supports_float() -> SupportsFloat:
    return "42"  # FN

  def returns_text() -> Text:
    return "Hello"  # OK

  Checkers = Tuple[Callable[[Expression], None], Callable[[Type], None]]
  def my_checkers() -> Checkers:
    return 1, 2

  def unknown_union() -> Union["GenericFixed", "Table"]:
    return TypeError("error")

  def my_str_union(typ: int) -> Union[str, np.dtype]:
    return str(typ)

  from typing import NoneType
  MyNone = NoneType
  def my_none() -> MyNone:
    return

def other_returns():
  def my_int(param) -> int:
    if param:
      return 42
    else:
      return "42"  # Noncompliant

  def my_int_returns_none() -> int:
    return  # Noncompliant

  def my_int_union_type_ok(cond) -> int:
    if cond:
      x = 42
    else:
      x = "hello"
    return x  # OK

  def my_list_union_nok(cond) -> List[str]:
    if cond:
      value = 42
    else:
      value = "hello"
    return value  # Noncompliant {{Return a value of type "list[str]" or update function "my_list_union_nok" type hint.}}

  def nested_func() -> int:
    def my_nested() -> str:
      return "hello"  # OK
    return 42  # OK

def functions_with_try_except():
  def my_int_try_except(cond) -> int:
    class A: ...
    try:
      x = "hello"
      x = A()
      return x  # Noncompliant {{Return a value of type "int" or update function "my_int_try_except" type hint.}}
    except: ...

  def my_list_union_nok(cond) -> List[str]:
    if cond:
      value = 42
    else:
      value = "hello"
    return value  # Noncompliant {{Return a value of type "list[str]" or update function "my_list_union_nok" type hint.}}

def custom_classes():
  class A: ...

  class B(A):
    def foo(): ...

  def my_func() -> A:
    return B()  # OK

  def my_func2() -> B:
    return A()  # Noncompliant


def generators():
  def my_generator() -> Generator:
    for i in range(10):
      yield i
    return "nothing left"  # OK, still a generator

  def my_generator_2() -> Generator[str]:
    for i in range(10):
      yield i  # Out of scope FN
    return "nothing left"  # OK, still a generator

  def not_a_generator(param) -> int:
    if param:
      return "hello" # Noncompliant {{Return a value of type "int" instead of "str" or update function "not_a_generator" type hint.}}
    else:
      yield 42  # Noncompliant {{Remove this yield statement or annotate function "not_a_generator" with "typing.Generator" or one of its supertypes.}}

  def empty_iterator() -> Iterator[None]:
    yield  # OK

  JsonDict = Dict[str, Any]
  def my_iterable_with_alias(chunks: Iterable[JsonDict]) -> Iterable[JsonDict]:
    for chunk in chunks:
      yield chunk  # OK

  def my_conditional_iterator(cond) -> Iterator[Tuple[object, object]]:
    if cond:
      return  # OK
    else:
      yield

  class MyIter:
    def __iter__(self):
      return iter("hello")

  def my_custom_iterable() -> MyIter:
#     ^^^^^^^^^^^^^^^^^^>     ^^^^^^>
    my_iter = MyIter()
    for elem in my_iter:
      yield elem  # Noncompliant {{Remove this yield statement or annotate function "my_custom_iterable" with "typing.Generator" or one of its supertypes.}}
#     ^^^^^^^^^^

def missing_return():
  """No issue if functions do not return as it might be due to custom error handling or a stub definition"""
  def raise_custom_error(message):
    raise ValueError(message)

  def get_int_might_raise_exception(x) -> int: # OK
    if x:
      return 42
    raise_custom_error("invalid input")

  def my_int_2(param) -> int:  # FN
    if param:
      return 42
    else:
      print("hello")

  def my_int_4() -> int:  # FN
    try:
      return 42
    except IndexError as e:
      print("hello")

  def not_implemented() -> int:
    return NotImplemented

  def my_stub() -> int:
    ...

def out_of_scope():
  def my_noreturn() -> NoReturn:
    """NoReturn functions should never return normally. To be checked with actual usage on Peach to avoid noise"""
    return None  # FN

  def my_list_oos() -> List[str]:
    return [1, 2, 3]  # FN

  def my_list_oos2() -> List[str]:
    return [1, "my_str", 2]  # OK

  def my_list_oos3() -> List[str]:
    x = 1
    y = 2
    return [x, y]  # FN

  def my_list_oos4() -> List[str]:
    x = 1
    y = 2
    my_list = [x, y]
    print("hello")
    my_list.append(3)
    return my_list  # FN

  def my_set_str() -> Set[str]:
    return {42, 43}  # FN

  def my_anystr(param) -> AnyStr:
    if param == 1:
      return "hello"
    elif param == 2:
      return u"foo"
    elif param == 3:
      return b"bar"
    elif param == 4:
      return bytes("hello")
    elif param == 5:
      return 42  # FN

  class Base:
    def meth(self): ...
  class C(Base):
    def another_meth(self): ...

  def declared_type(param: Base) -> Optional[C]:
    if isinstance(param, C):
      return param # OK
    return None


def type_dict():
  class MyCustomDict(TypedDict):
    user_ids: Set[int]
    message_ids: Set[int]

  def my_dict() -> MyCustomDict:
    users = {1,2,3}
    messages = {1,2,3}
    return dict(user_ids=users, message_ids=messages)

async def no_issue_for_async_iterators() -> AsyncIterator[int]:
    yield 1

async def no_issue_for_async_generators() -> AsyncGenerator[int]:
    yield 1

async def async_function_returning_iterator() -> Iterator[int]:
    yield 1 # Noncompliant {{Annotate function "async_function_returning_iterator" with "typing.AsyncGenerator" or one of its supertypes.}}

def echo_round() -> Generator[int, float, str]:
  sent = 0
  while sent < 3:
    sent += 1
    x = yield sent
  return 'Done'
