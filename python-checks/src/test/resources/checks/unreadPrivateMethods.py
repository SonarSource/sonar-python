class A:
  __foo = 42 # OK, raised by S4487

  def _unused_single_underscore(self): ... # OK

  def __unused(self): ... # Noncompliant {{Remove this unused class-private '__unused' method.}}
#     ^^^^^^^^

  @classmethod
  def __unused_cls_method(cls): ... # Noncompliant

  @staticmethod
  def __unused_static_method(): ... # Noncompliant

  def __used(self): ...

  @classmethod
  def __used_cls_method(cls): ...

  @staticmethod
  def __used_static_method(): ...

  def __used_in_cls_body(): ...

  __used_in_cls_body()

  def __init__(self):
    print(self.__used())
    print(A.__used_cls_method())
    print(A.__used_static_method())

  def __used_through_not_self(self):
    pass

  def some_usage(self, node):
    node.__used_through_not_self()

def method_decorator(func):
  def inner(*args, **kwargs):
    return func(*args, **kwargs)
  return inner

class ClassWithMethodDecorator:
  @method_decorator
  def __getVariable(self):
    return self._v

  @classmethod
  def __setVariable(self, v): # Noncompliant
    self._v = v

  def __printVariable(self): # Noncompliant
    print(self._v)

def class_decorator(c):
  return c

@class_decorator
class ClassWithDecorator:

  def __printVariable(self):
    print(self.__v)
