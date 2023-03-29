import typing
from typing import List, Set, Tuple, Mapping, Sequence, Iterable, Type, AbstractSet, Callable, NewType
from typing import Dict as MyDict
from typing import FrozenSet as FSet

import collections
from collections.abc import Iterable as ABCIterable, Sequence as ABCSequence, Mapping as ABCMapping, Set as ABCSet, Callable as ABCCallable

def foo(p: typing.Tuple[int, str]): # Noncompliant {{Use the built-in generic type `tuple` instead of its typing counterpart.}}
          #^^^^^^^^^^^^^^^^^^^^^^
    pass

def foobar() -> typing.Dict[str, int]: # Noncompliant {{Use the built-in generic type `dict` instead of its typing counterpart.}}
               #^^^^^^^^^^^^^^^^^^^^^
    pass

def nested() -> dict[str, list[FSet[str]]]: # Noncompliant {{Use the built-in generic type `frozenset` instead of its typing counterpart.}}
                              #^^^^^^^^^
    pass

def with_var() -> typing.Set[str]: # Noncompliant {{Use the built-in generic type `set` instead of its typing counterpart.}}
                 #^^^^^^^^^^^^^^^
    my_var: typing.FrozenSet[int] = None # Noncompliant
           #^^^^^^^^^^^^^^^^^^^^^
    pass

class Bar:
    class_var: typing.List[str] # Noncompliant
              #^^^^^^^^^^^^^^^^

    def foo(p: typing.List[int]): # Noncompliant {{Use the built-in generic type `list` instead of its typing counterpart.}}
              #^^^^^^^^^^^^^^^^
        pass

    def foobar() -> typing.Dict[str, list[Tuple[str, int]]]: # Noncompliant 2
        pass

    def with_var() -> typing.Set[str]: # Noncompliant
                     #^^^^^^^^^^^^^^^
        my_var: typing.FrozenSet[int] = None # Noncompliant
               #^^^^^^^^^^^^^^^^^^^^^
        pass

class No_Module:
    class_var: List[str] # Noncompliant
              #^^^^^^^^^

    def foo(p: Tuple[int, ...]): # Noncompliant
              #^^^^^^^^^^^^^^^
        pass

    def foobar() -> MyDict[str, int]: # Noncompliant {{Use the built-in generic type `dict` instead of its typing counterpart.}}
                   #^^^^^^^^^^^^^^^^
        pass

    def with_var() -> Set[str]: # Noncompliant
                     #^^^^^^^^
        my_var: FSet[int] = None # Noncompliant
               #^^^^^^^^^
        pass

class Success:
    class_var: list[str]

    def foo(p: tuple[int]):
        pass

    def foobar() -> dict[str, tuple[str, ...]]:
        pass

    def with_var() -> set[str]:
        my_var: frozenset[int] = None
        pass


class CheckCollections:
    class_var: typing.Iterable[str] # Noncompliant {{Use the built-in generic type `collections.abc.Iterable` instead of its typing counterpart.}}
              #^^^^^^^^^^^^^^^^^^^^

    def foo(p: typing.Sequence[int]): # Noncompliant {{Use the built-in generic type `collections.abc.Sequence` instead of its typing counterpart.}}
              #^^^^^^^^^^^^^^^^^^^^
        pass

    def foobar() -> typing.Mapping[str, int]: # Noncompliant {{Use the built-in generic type `collections.abc.Mapping` instead of its typing counterpart.}}
                   #^^^^^^^^^^^^^^^^^^^^^^^^
        pass

    def with_var(m:Mapping[str, int]) -> Sequence[str]: # Noncompliant 2
        my_var: Iterable[int] = None # Noncompliant
               #^^^^^^^^^^^^^
        pass

    def foo_set(my_set: typing.AbstractSet[int]):# Noncompliant {{Use the built-in generic type `collections.abc.Set` instead of its typing counterpart.}}
                       #^^^^^^^^^^^^^^^^^^^^^^^
        pass

    def call():
        some_var: Callable[[int, str], list[int]] # Noncompliant {{Use the built-in generic type `collections.abc.Callable` instead of its typing counterpart.}}
                 #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class SuccessCollections:
    class_var: collection.abc.Iterable[str]

    def foo(p: collection.abc.Sequence[bool]):
        pass

    def foobar() -> collection.abc.Mapping[str, int]:
        pass

    def with_var(m:ABCMapping[str, int]) -> ABCSequence[str]:
        my_var: ABCIterable[int] = None
        pass

    def foo_set(my_set: ABCSet[int]):
        pass

    def call():
        some_var: ABCCallable[[int, str], list[int]]

class GenericType:
    class_var: typing.Type[str] # Noncompliant
              #^^^^^^^^^^^^^^^^

    def foo(p: Type[int]): # Noncompliant {{Use the built-in generic type `type` instead of its typing counterpart.}}
              #^^^^^^^^^
        pass

    def nested(m:ABCMapping[Type[str], int]): # Noncompliant
                           #^^^^^^^^^
        pass

    def success():
        my_var: type[int] = None
        pass
