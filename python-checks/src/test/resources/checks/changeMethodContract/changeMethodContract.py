from changeMethodContractParent import ParentClass, IntermediateWithGenericParentClass, IntermediateClass
from typing import Any

class ChildClass(ParentClass):
  def __init__(p1, p2, p3): ... # OK
  def compliant(self, param1): ...
  def with_keyword_only(self, param1, param2, param3): ... # Noncompliant
  def with_decorator(self): ... # OK
  def with_default(self, param1=1): ... # OK
  def my_method(self, param1, param2): ... # Noncompliant {{Remove parameter param2 or provide default value.}}
#                             ^^^^^^
  def with_kwargs(self, param1): ... # OK
  def __private_method(self, param1): ... # Noncompliant
  def attr(self): ...
  def using_tuple(self, (a, b, c)): ... # FN

class ChildWithGenericParentClass(ParentClass[Any]):
  def __init__(p1, p2, p3): ... # OK
  def compliant(self, param1): ...
  def with_keyword_only(self, param1, param2, param3): ... # Noncompliant
  def with_decorator(self): ... # OK
  def with_default(self, param1=1): ... # OK
  def my_method(self, param1, param2): ... # Noncompliant {{Remove parameter param2 or provide default value.}}
  #                           ^^^^^^
  def with_kwargs(self, param1): ... # OK
  def __private_method(self, param1): ... # Noncompliant
  def attr(self): ...
  def using_tuple(self, (a, b, c)): ... # FN

class ChildFromIntermediateClass(IntermediateClass):
  def __init__(p1, p2, p3): ... # OK
  def compliant(self, param1): ...
  def with_keyword_only(self, param1, param2, param3): ... # Noncompliant
  def with_decorator(self): ... # OK
  def with_default(self, param1=1): ... # OK
  def my_method(self, param1, param2): ... # Noncompliant {{Remove parameter param2 or provide default value.}}
  #                           ^^^^^^
  def with_kwargs(self, param1): ... # OK
  def __private_method(self, param1): ... # Noncompliant
  def attr(self): ...
  def using_tuple(self, (a, b, c)): ... # FN

class ChildFromIntermediateWithGenericParentClass(IntermediateWithGenericParentClass):
  def __init__(p1, p2, p3): ... # OK
  def compliant(self, param1): ...
  def with_keyword_only(self, param1, param2, param3): ... # Noncompliant
  def with_decorator(self): ... # OK
  def with_default(self, param1=1): ... # OK
  def my_method(self, param1, param2): ... # Noncompliant
  def with_kwargs(self, param1): ... # OK
  def __private_method(self, param1): ... # Noncompliant
  def attr(self): ...
  def using_tuple(self, (a, b, c)): ... # FN

class MoreThanOneExtra(ParentClass):
  def my_method(self, param1, param2, param3): ... # Noncompliant 2
  def compliant(self, param1, (a, b)): ... # FN

class LessParams(ParentClass):
  def my_method(self): ... # Noncompliant {{Add missing parameters param1.}}
#     ^^^^^^^^^
  def with_default(self): ... # Noncompliant

class NoParams(ParentClass):
  def my_method(): ... # Noncompliant {{Add missing parameters self param1.}}

class WithDefaultRemoved(ParentClass):
  def with_default(self, param1): ... # Noncompliant {{Add a default value to parameter param1.}}

class ChangedParamOrder(ParentClass):
  def changed_param_order(self, param2, param1): ... # Noncompliant {{Move parameter param1 to position 1.}} {{Move parameter param2 to position 2.}}

class ChangedParamName(ParentClass):
  def my_method(self, paramX): ...

class ExtraParamWithDefault(ParentClass):
  def my_method(self, param1, param2=42): ... # OK

class UsingDecorator(ParentClass):
  @foo
  def my_method(self): ...

class A:
    def my_method(self): ...

def my_method(self): ... # OK

from mod import OtherClass

class UnresolvedParent(OtherClass):
  def my_method(self): ... # OK

class MyString(str):
  def capitalize(self, p1): ... # Noncompliant {{Remove parameter p1 or provide default value. This method overrides str.capitalize.}}


class Intermediate(ParentClass): ...

class TransitivelyOverriding(Intermediate):
  def my_method(self): ... # Noncompliant
  def compliant(self, param1): ...

class KeywordOnlyParameters(ParentClass):
  def my_method(self, param1, *, param2): ... # Noncompliant
#                                ^^^^^^

class PositionalOnlyParameters(ParentClass):
  def my_method(self, param1, /, param2): ... # Noncompliant
#                                ^^^^^^

class KeywordOnlyOneExtra(ParentClass):
  def with_keyword_only(self, param1, *, param2, param3): ... # Noncompliant

class PreviouslyKeywordOrPositional(ParentClass):
    def with_keyword_only(self, *, param1, param2): ... # Noncompliant {{Make parameter param1 keyword-or-positional.}}
#                                  ^^^^^^

class PositionalOnly(ParentClass):
    def positional_only(self, param1, unknown, /, param2, *, param3): ... # Noncompliant {{Change this method signature to accept the same arguments as the method it overrides.}}
#       ^^^^^^^^^^^^^^^

class ChildClassPosOnlyMovedBad1(ParentClass):
    def positional_only(self, param1, param2, /, *, param3): ... # Noncompliant {{Make parameter param2 keyword-or-positional.}}
#                                     ^^^^^^

class ChildClassKeywordOnlyToKworPos(ParentClass):
    def positional_only(self, param1, /, param2, param3): ... # OK. Keyword arguments are still accepted.

class ChildClassKeywordOnlyToPosOnly(ParentClass):
    def positional_only(self, param1, param2, param3, /): ... # Noncompliant {{Make parameter param2 keyword-or-positional.}} {{Make parameter param3 keyword only.}}
class ChildClassPosOnlyToKworPos(ParentClass):
    def positional_only(self, param1, param2,*, param3): ... # OK The substitution is correct

class ChildClassReorderingKW(ParentClass):
    def with_two_keyword_only(self, *, param2, param1): ...  # OK. Reordering keyword only parameters is ok


class ChildClassReorderingAndExtra(ParentClass):
    def my_method(self, inserted, param1): ... # Noncompliant
#                       ^^^^^^^^

class ChildClassOneExtraDefault(ParentClass):
    def my_method(self, foo, other=42): ... # OK

from io import BufferedIOBase, BytesIO, TextIOWrapper

class StreamingBuffer(BytesIO):
  def read(self): ... # Noncompliant {{Add 1 missing parameter. This method overrides _io._BufferedIOBase.read.}}

class MyTextIOWrapper(TextIOWrapper):
  def seek(self): ... # Noncompliant {{Add 2 missing parameters. This method overrides _io.TextIOWrapper.seek.}}

import datetime

class MyTZInfo(datetime.tzinfo):
  def tzname(self): # Noncompliant
    ...


class MyDictionary(dict):
    def get(self, key):
        ...

from abc import abstractmethod
import abc

class AbstractSuperclass:
  @abstractmethod
  def foo(self, a: int):
    ...

  @abc.abstractmethod
  def bar(self, a: int):
    ...

class InheritedFromAbstractSuperclass(AbstractSuperclass):
  def foo(self): # Noncompliant
    ...

  def bar(self): # Noncompliant
    ...


class PositionalOnlyContract:
    def positional(self, first, second, /): ...
    def optional(self, value=0, /): ...
    def required(self, value, /): ...
    def mixed(self, x, /, y): ...

class RenamedPositionalContract(PositionalOnlyContract):
    def positional(self, second, first, /): ...
    def optional(self, renamed): ...  # Noncompliant {{Add a default value to parameter renamed.}}
    def required(self, renamed, value): ...  # Noncompliant {{Remove parameter value or provide default value.}}
    def mixed(self, y, x): ...  # Noncompliant {{Move parameter y to position 2.}}

class ExtendedPositionalContract(PositionalOnlyContract):
    def positional(self, second, first, extra=None): ...
    def optional(self, renamed, extra): ...  # Noncompliant {{Add a default value to parameter renamed.}} {{Remove parameter extra or provide default value.}}
    def required(self, renamed, *, value): ...  # Noncompliant {{Remove parameter value or provide default value.}}
    def mixed(self, renamed, y): ...

class WidenedPositionalContract(PositionalOnlyContract):
    def positional(self, second, first): ...
    def optional(self, renamed=0, extra=None): ...
    def required(self, renamed, value=0): ...

class NarrowedPositionalContract(PositionalOnlyContract):
    def optional(self, *, renamed): ...  # Noncompliant {{Add a default value to parameter renamed.}} {{Make parameter renamed accept positional arguments.}}
#                         ^^^^^^^
    def required(self, *, renamed, extra): ...  # Noncompliant {{Make parameter renamed accept positional arguments.}} {{Remove parameter extra or provide default value.}}

# Stub receivers may lack the positional-only flag carried by later parameters.

class OptionalExtraOnStubContract(BufferedIOBase):
    def read1(self, n, extra=None): ...  # Noncompliant {{Add a default value to parameter n. This method overrides _io._BufferedIOBase.read1.}}

class RequiredExtraOnStubContract(BufferedIOBase):
    def read1(self, n, extra): ...  # Noncompliant {{Add a default value to parameter n. This method overrides _io._BufferedIOBase.read1.}} {{Remove parameter extra or provide default value. This method overrides _io._BufferedIOBase.read1.}}


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Model as ModelAlias
from tensorflow.keras.layers import Layer

# Keras inspects custom signatures; its library call hooks accept *args/**kwargs.
class CustomModel(Model):
    def call(self, inputs): ...

class MultiInputModel(ModelAlias):
    def call(self, left, right, *, training=False, mask=None): ...

class CustomLayer(Layer):
    def call(self, inputs): ...
    def build(self): ...  # Noncompliant {{Add 1 missing parameter. This method overrides tensorflow.keras.layers.Layer.build.}}

class MultiInputLayer(tf.keras.layers.Layer):
    def call(self, left, right): ...

class RequiredTrainingModel(tf.keras.Model):
    def call(self, inputs, training, extra=None): ...

# Source-defined contracts on Keras subclasses remain checked.
class ChangedUserModel(CustomModel):
    def call(self): ...  # Noncompliant {{Add missing parameters inputs.}}

class ChangedUserLayer(MultiInputLayer):
    def call(self, left): ...  # Noncompliant {{Add missing parameters right.}}

class OrdinaryCallParent:
    def call(self, inputs, training=False, mask=None): ...

class OrdinaryCallChild(OrdinaryCallParent):
    def call(self, inputs): ...  # Noncompliant {{Add missing parameters training mask.}}
