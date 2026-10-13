from __future__ import annotations
from typing import List, final
class A: ...

def custom(param1: A, param2: A):
  param1 + param2 # Noncompliant {{Fix this "+" operation; Previous type checks suggest that operands have incompatible types (A and A).}}
  #      ^
  -param1 # Noncompliant {{Fix this "-" operation; Previous type checks suggest that operand has incompatible type.}}

def builtin(param1: int, param2: str, param3: List[str]):
  param1 + param2 # Noncompliant {{Fix this "+" operation; Previous type checks suggest that operands have incompatible types (int and str).}}
  param1 + param3 # Noncompliant


from unittest.mock import Mock, MagicMock


# We should not raise any issues on mocks as they could be monkey patched to fit any types
def mocks(mock: Mock, magic_mock:MagicMock):
    mock + 3 # Ok
    -magic_mock # Ok


class MockExtention(Mock):
    ...


def custom_mock(extended_mock: MockExtention):
  extended_mock + 3  # Ok
  -extended_mock


def add_operations(cls):
    cls.__add__ = lambda self, other: 42
    cls.__neg__ = lambda self: 42
    return cls


@add_operations
class Decorated:
    def run(self: Decorated):
        self + self
        -self


class Child(Decorated):
    def run(self: Child):
        self + self
        -self


@final
class Marked:
    def run(self: Marked):
        self + self  # Noncompliant
        -self  # Noncompliant


class Plain:
    def run(self: Plain):
        self + self  # Noncompliant
        -self  # Noncompliant
