def func(p1, p2, p3, p4): ...  # Noncompliant
#        ^^^^^^^^^^^^^^

class ParentClass:
  def __init__(self, p1, p2, p3, p4): ... # Noncompliant

class Wrapper(ParentClass):
  def __init__(self, p1, p2, p3, p4, p5): ...  # OK (parent is already non compliant)

  def readline(self, __size: int = ..., p2, p3, p4) -> str: ...  # Noncompliant

  def new_method_ok(self, p1, p2, *, p3) -> str: ...

  def new_method_nok(self, p1, p2, p3, p4) -> str: ...  # Noncompliant

class MyOtherWrapper(Wrapper): ...

class ChildWithComplexHierarchy(MyOtherWrapper):
  def __init__(self, p1, p2, p3, p4, p5, p6): ...

class SuperBase:
  def method_nok(self, p1, p2, p3, p4): ...  # Noncompliant

  @staticmethod
  def my_static_method(p1, p2, p3): ... # OK

class Base(SuperBase):

  @staticmethod
  def my_static_method(p1, p2, p3): ... # OK

class Child(Base):
  def method_nok(self, p1, p2, p3, p4): ...  # OK

  @staticmethod
  def my_static_method(p1, p2, p3, p4): ... # Noncompliant
