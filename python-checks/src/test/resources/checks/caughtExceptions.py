from typing import Tuple, Type, Union


class CustomException:
  ...

class MyException(Exception):
  ...
def foo():
  try:
    ...
  except CustomException: # Noncompliant {{Change this expression to be a class deriving from BaseException or a tuple of such classes.}}
  #      ^^^^^^^^^^^^^^^
    ...
  except (None, BaseException):  # Noncompliant
  #       ^^^^
    ...

  l = [TypeError, ValueError]

  try:
    ...
  except [TypeError, ValueError]:  # Noncompliant
  #      ^^^^^^^^^^^^^^^^^^^^^^^
    ...
  except {TypeError, ValueError}:  # Noncompliant
  #      ^^^^^^^^^^^^^^^^^^^^^^^
    ...
  except l: # Noncompliant
    ...

  try:
     ...
  except MyException:
    ...
  except (MyException):
    ...
  except MyException if True else BaseException:
    ...
  except OSError:
    ...
  except unknown:
    ...
  except ArithmeticError:
    ...
  except:
    ...

  x = 42

  try:
    ...
  except x: # Noncompliant
    ...

  except int: # Noncompliant
  #      ^^^
    ...

  exceptions = (ZeroDivisionError, OverflowError)
  try:
    foo = 1/0
  except exceptions:
    foo = 0

  import smtplib
  try:
    ...
  except smtplib.SMTPServerDisconnected:  # OK, SMTPServerDisconnected when Python version is unknown
    ...

  incorrect_exception_alias = CustomException
  try:
    ...
  except incorrect_exception_alias:  # FN SONARPY-1666
    ...

def exception_type_parameter(exception_type: Union[Type[Exception], Tuple[Type[Exception], ...]]):
  try:
    ...
  except exception_type:
    ...


def non_exception_parameter(e: int):
  try:
    ...
  except e: # Noncompliant
    ...


def valid_mixed_union_handler(flag):
  errors = ValueError if flag else (TypeError,)
  try:
    ...
  except errors:
    ...


def invalid_union_parameter(errors: int | str):
  try:
    ...
  except errors: # Noncompliant
    ...


def invalid_multi_binding_union(flag):
  if flag:
    errors = 42
  else:
    errors = "invalid"
  try:
    ...
  except errors: # Noncompliant
    ...


def except_star():
  try:
    ...
  except* CustomException: # Noncompliant {{Change this expression to be a class deriving from BaseException or a tuple of such classes.}}
    ...
  except* (None, BaseException):  # Noncompliant
    ...
  except* list():  # Noncompliant
    ...


def reassigned_exception():
    my_exception = None
    my_exception = ZeroDivisionError
    try:
        ...
    except my_exception:
        ...


def except_type():
    try:
        raise smth
    except type(smth) as e:
        ...
