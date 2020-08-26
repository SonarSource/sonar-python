from io import TextIOWrapper

def func(p1, p2, p3, p4): ...  # Noncompliant
#        ^^^^^^^^^^^^^^

class MyTextIOWrapper(TextIOWrapper):
  # FP (ambiguous symbol in type hierarchy)
  def __init__(
          self,  # Noncompliant
          buffer: IO[bytes],
          encoding: Optional[str] = ...,
          errors: Optional[str] = ...,
          newline: Optional[str] = ...,
          line_buffering: bool = ...,
          write_through: bool = ...,
      ) -> None: ...  # OK (parent is already non compliant)

  def readline(self, __size: int = ..., p2, p3, p4) -> str: ...  # Noncompliant

  def new_method_ok(self, p1, p2, *, p3) -> str: ...

  def new_method_nok(self, p1, p2, p3, p4) -> str: ...  # Noncompliant

class MyOtherTextIOWrapper(TextIOWrapper): ...

class ChildWithComplexHierarchy(MyOtherTextIOWrapper):
  # FP (ambiguous symbol in type hierarchy)
  def __init__(
          self,  # Noncompliant
          buffer: IO[bytes],
          encoding: Optional[str] = ...,
          errors: Optional[str] = ...,
          newline: Optional[str] = ...,
          line_buffering: bool = ...,
          write_through: bool = ...,
      ) -> None: ...

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
