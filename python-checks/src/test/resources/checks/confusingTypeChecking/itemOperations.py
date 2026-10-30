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


class DictionaryDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return {}

class DescriptorOwner:
    items: DictionaryDescriptor = DictionaryDescriptor()
    ordinary: int = 1

    def update(self):
        self.items["key"] = 1
        _ = self.items["key"]
        del self.items["key"]
        (self.items)["key"] = 1
        self.ordinary["key"] = 1  # Noncompliant


def descriptor_object_itself(value: DictionaryDescriptor):
    value["key"] = 1  # Noncompliant


def class_descriptor_read():
    DescriptorOwner.items["key"] = 1  # Noncompliant

def unknown_descriptor_receiver(value):
    value.items["key"] = 1

class IntegerDescriptor:
    def __get__(self, instance, owner) -> int:
        return 1

class KnownDescriptorOwner:
    number: IntegerDescriptor = IntegerDescriptor()

    def update(self):
        self.number["key"] = 1  # Noncompliant


def decorate_getter(function):
    return lambda *args: {}

class DecoratedDescriptor:
    @decorate_getter
    def __get__(self, instance, owner) -> int:
        return 1

class AsyncDescriptor:
    async def __get__(self, instance, owner) -> int:
        return 1

class DictionaryReturnDescriptor:
    def __get__(self, instance, owner) -> dict:
        return {}

class MixedReturnDescriptor:
    def __get__(self, instance, owner) -> int | dict:
        ...

class IncompatibleReturnDescriptor:
    def __get__(self, instance, owner) -> int | float:
        ...

class DescriptorReturnTypes:
    decorated: DecoratedDescriptor = DecoratedDescriptor()
    asynchronous: AsyncDescriptor = AsyncDescriptor()
    dictionary: DictionaryReturnDescriptor = DictionaryReturnDescriptor()
    mixed: MixedReturnDescriptor = MixedReturnDescriptor()
    incompatible: IncompatibleReturnDescriptor = IncompatibleReturnDescriptor()

    def update(self):
        self.decorated["key"] = 1
        self.asynchronous["key"] = 1  # Callable effects remain unknown.
        self.dictionary["key"] = 1
        self.mixed["key"] = 1
        self.incompatible["key"] = 1  # Noncompliant


from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

class MappedDictionary:
    data: Mapped[dict] = mapped_column(JSON)

    def update(self, **kwargs):
        self.data = kwargs
        self.data["prefix"] = "value"

    def update_later(self):
        self.data["key"] = "value"
