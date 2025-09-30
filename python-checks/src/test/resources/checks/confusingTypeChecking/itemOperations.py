from typing import List

class A: ...
#     ^> {{Definition of "A".}}

def custom(param: A):
  param[42] # Noncompliant {{Fix this "__getitem__" operation; Previous type checks suggest that "param" does not have this method.}}
# ^^^^^^^^^
  param[42] = 42 # Noncompliant {{Fix this "__setitem__" operation; Previous type checks suggest that "param" does not have this method.}}
  del param[42] # Noncompliant {{Fix this "__delitem__" operation; Previous type checks suggest that "param" does not have this method.}}

def builtin(param1: memoryview, param2: frozenset, param3: List[int]):
  del param1[0]  # Noncompliant
  param2[42] = 42 # Noncompliant
  param3[0]

def derived(param1: int, param2: int, *param3: int):
  (param1 + param2)[0] # Noncompliant {{Fix this "__getitem__" operation; Previous type checks suggest that this expression does not have this method.}}
  param3[42] # OK

def f(val: object):
  if not isinstance(val, tuple):
    ...
  val[0]

def checked_with_in(obj: object, other: object, prop):
  if prop in obj:
    obj[prop]
  if other in obj:
    other[prop] # Noncompliant

def checked_with_not_in(obj: object, prop):
  if prop not in obj:
    return
  obj[prop]


from unittest.mock import Mock, MagicMock


# We should not raise any issues on mocks as they could be monkey patched to fit any types
def mocks(mock:Mock, magic_mock: MagicMock):
    del mock[42]
    mock[42] = 42
    a = mock[42]
    magic_mock[42] = 42


class MockExtention(Mock):
    ...


def custom_mock(extended_mock: MockExtention):
  del extended_mock[42]
  extended_mock[42] = 42
  a = extended_mock[42]

from torch.utils.data import Dataset, IterableDataset, TensorDataset, StackDataset, ConcatDataset, ChainDataset, Subset
def torch_datasets(d1:Dataset, d2:IterableDataset, d3:TensorDataset, d4:StackDataset, d5:ConcatDataset, d6:ChainDataset, d7:Subset):
  d1[42]
  d2[42]
  d3[42]
  d4[42]
  d5[42]
  d6[42]
  d7[42]
