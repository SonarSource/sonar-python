class ParentClass(object):
  def __init__(p1, p2): ...
  def compliant(self, param1): ...
  def my_method(self, param1): ...
  def with_default(self, param1=42): ...
  def changed_param_order(self, param1, param2): ...
  def with_keyword_only(self, param1, *, param2): ...
  def with_kwargs(self, param1, **kwargs): ...
  @foo
  def with_decorator(self, param1): ...
  def __private_method(self): ...
  attr = 42

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

class MoreThanOneExtra(ParentClass):
  def my_method(self, param1, param2, param3): ... # Noncompliant 2

class LessParams(ParentClass):
  def my_method(self): ... # Noncompliant {{Add missing parameters param1.}}
#     ^^^^^^^^^
  def with_default(self): ... # OK

class NoParams(ParentClass):
  def my_method(): ... # Noncompliant {{Add missing parameters self param1.}}

class WithDefaultRemoved(ParentClass):
  def with_default(self, param1): ... # Noncompliant {{Add a default value to parameter param1.}}

class ChangedParamOrder(ParentClass):
  def changed_param_order(self, param2, param1): ... # Noncompliant {{Move parameter param1 to position 1.}} {{Move parameter param2 to position 2.}}

class ChangedParamName(ParentClass):
  def my_method(self, paramX): ... # Noncompliant {{Rename this parameter as "param1".}}
#                     ^^^^^^

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
  def capitalize(self, p1): ... # Noncompliant


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
