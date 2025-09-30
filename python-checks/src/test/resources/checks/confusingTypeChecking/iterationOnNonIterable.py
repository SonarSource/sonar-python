from typing import Union

class A: ...
#     ^> {{Definition of "A".}}

class NewStyleIterable:
  def __iter__(self): ...

class OldStyleIterable:
  def __getitem__(self): ...

def my_integer() -> int:
  ...

class MyAsyncIterable:
  def __aiter__(self): ...

def custom(param1: A, param2: NewStyleIterable, param3: OldStyleIterable):
  a, b, c = *param1  # Noncompliant {{Replace this expression; Previous type checks suggest that "param1" has type "A" and isn't iterable.}}
#            ^^^^^^
  a, b, c = *param2
  a, b, c = *param3

def builtin(param1: int, param2: float):
  for x in param1:  # Noncompliant {{Replace this expression; Previous type checks suggest that "param1" has type "int" and isn't iterable.}}
    ...
  for y in param2:  # Noncompliant {{Replace this expression; Previous type checks suggest that "param2" has type "float" and isn't iterable.}}
    ...
  not_an_iterable: int = 42
  print(*not_an_iterable)  # FN
  my_var = my_integer()
  print(*my_var)  # Noncompliant {{Replace this expression; Previous type checks suggest that "my_var" has type "int" and isn't iterable.}}
  print(*my_integer())  # Noncompliant {{Replace this expression; Previous type checks suggest that it has type "int" and isn't iterable.}}


def union_types(param1: Union[int, float], param2: Union["unknown", "unknown2"]):
  for x in param1:  # Noncompliant {{Replace this expression; Previous type checks suggest that "param1" has type "Union[int, float]" and isn't iterable.}}
    ...
  for x in param2:  # OK
    ...

def async_iterable(param1: MyAsyncIterable):
  for x in param1:  # Noncompliant {{Add "async" before "for"; Previous type checks suggest that "param1" has type "MyAsyncIterable" and is an async generator.}}
    ...

from unittest.mock import Mock, MagicMock


# We should not raise any issues on mocks as they could be monkey patched to fit any type
def mocks_no_issue(mock: Mock,magic_mock: MagicMock):
  a, *rest = mock
  iter(mock)
  for elem in magic_mock: ... # OK


class MockExtention(Mock):
    ...


def custom_mock(extended_mock: MockExtention):
  a, *rest = extended_mock
  iter(extended_mock)
  for elem in extended_mock: ... # OK

from torch.utils.data import Dataset, IterableDataset, TensorDataset, StackDataset, ConcatDataset, ChainDataset, Subset
def torch_iterable_dataset(d:IterableDataset):
  for elem in d: ... # OK