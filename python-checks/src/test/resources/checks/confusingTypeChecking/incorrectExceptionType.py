from typing import Union, Optional
class A: ...
class CustomException(BaseException): ...
class AnotherException(CustomException): ...

def my_int() -> int:
  ...

def custom(param1: A, param2: CustomException, param3: AnotherException):
  raise param1  # Noncompliant {{Fix this "raise" statement; Previous type checks suggest that "param1" has type "A" and is not an exception.}}
# ^^^^^^^^^^^^
  raise param2
  raise param3


def builtin(param1: str, param2: Exception, param3: BaseException, param4: ValueError, param5: Union[BaseException, str], param6: Optional[BaseException]):
  raise param1  # Noncompliant {{Fix this "raise" statement; Previous type checks suggest that "param1" has type "str" and is not an exception.}}
  raise param2
  raise param3
  raise param4
  raise my_int()  # Noncompliant {{Fix this "raise" statement; Previous type checks suggest that this expression has type "int" and is not an exception.}}
  raise param5
  raise param6
