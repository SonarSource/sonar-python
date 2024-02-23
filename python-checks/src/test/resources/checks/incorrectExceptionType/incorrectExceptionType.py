from module import usage
from mod2 import pb
from mod2 import ImportedError
import external

class SomeClass:
  pass

class AnotherClass(object):
  pass

class SomeError(BaseException):
  pass

class SomeDerivedError(SomeError):
  pass

class DefinedError(pb.Error):
  pass

class ClassWithCallExpression(foo()):
  pass

def raise_string_literal():
  raise "something went wrong" # Noncompliant

def raise_class_with_call_expression_arg():
  raise ClassWithCallExpression

def raise_child_of_imported_type():
  raise DefinedError()

def raise_imported_error():
  raise ImportedError()

def func():
  err_dict = {1: (BaseException, 'error message 1'), 2: (BaseException, 'error message 2')}
  e, msg = err_dict[0]
  raise e(msg)

def raise_builtin_exception_type():
  raise NotImplementedError() # OK

def raise_builtin_exception_type_python2():
  raise StandardError() # OK

def raise_builtin_constant():
  raise NotImplemented # Noncompliant {{Change this code so that it raises an object deriving from BaseException.}}
# ^^^^^^^^^^^^^^^^^^^^

def raise_builtin_function():
  raise object() # Noncompliant

def incorrect_usage_of_NotImplemented():
  raise NotImplemented("foo") # Noncompliant {{Change this code so that it raises an object deriving from BaseException.}}
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

def raise_BaseException_type():
  raise SomeError() # OK

def raise_BaseException_class():
  raise SomeError

def raise_non_exception_class():
  raise SomeClass # Noncompliant

def raise_unknown_type():
  raise UnknownType() # OK: unknown symbol will be reported by S3827

def raise_regular_class():
  raise SomeClass() # Noncompliant

def raise_regular_class_child():
  raise AnotherClass() # Noncompliant

def raise_exception_child():
  raise SomeDerivedError() # OK

def empty_raise():
  raise # OK

def raise_imported_type():
  raise usage.UsageError("error")

def raise_from_variable():
  e = Whatever()
  raise e # Not handled yet

def raise_module_attribue():
  raise __package__ # Noncompliant

class Clazz(object):
  def get_some_error(self):
          return SomeError()

  def raise_from_method_call(self):
    raise self.get_some_error()

def raise_from_external_import_with_same_name_as_builtin():
  raise external.next() # OK

smth = SomeClass, AnotherClass
class SomeUnpacked(*smth):
  pass

def fun():
  raise SomeUnpacked()

# Symbol with multiple bindings
if p:
  class MultipleBindings: pass
else:
  class MultipleBindings(BaseException): pass

raise MultipleBindings() # OK

# Type inference
def raise_with_type_inference():
  a = SomeClass()
  raise a # Noncompliant

def union_type():
  a = SomeClass()
  a = AnotherClass()
  raise a # Noncompliant

def raise_with_type_inference():
  a = SomeDerivedError()
  raise a # OK

def raise_with_type_inference_fn():
  a = AnotherClass
  raise a # FN
  e = "hello "
  raise e # Noncompliant

def raise_with_str_concat():
  a = AnotherClass
  raise a # FN
  e = "hello "
  raise e + "world" # Noncompliant

def raise_builtin_exception_with_fqn_null():
  raise IOError()


def no_fp_on_nonlocal_variables(x):
    exception = None
    def do_something():
        nonlocal exception
        if foo():
            exception = ValueError("Hello")
    do_something()
    if exception:
        raise exception


def reassigned_exception():
    my_exception = None
    my_exception = ValueError
    try:
        ...
    except my_exception:
        raise my_exception
