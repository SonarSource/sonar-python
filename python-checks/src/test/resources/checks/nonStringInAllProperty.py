from external import unknown_imported

class MyClass:
  @property
  def my_name(): ...
  @staticmethod
  def my_staticmethod(): ...
  @overload
  def my_overloaded_stub(): ...
  @overload
  def my_overloaded_stub(): ...

class MyStr(str): ...
def my_func(): ...

def within_a_function():
  __all__ = [MyClass, "MyClass"] # OK, not module level
  global my_global_var
  my_global_var = "some string"

__all__ = MyClass # Bug but out of scope
__all__ = "MyClass" # Bug but out of scope
__all__ = ["foo", "bar"] # OK
__all__ = [MyClass, "bar"] # Noncompliant
#          ^^^^^^^

__all__ = ("foo", "bar") # OK
__all__ = (MyClass, "bar") # Noncompliant
#          ^^^^^^^
__all__, b = "foo", "bar" # Out of scope
__all__ = b = (MyClass, "bar") # Noncompliant

a = MyClass()
a.b = 42
var = 42
reassigned_var = var
reassigned_class = a
my_global_var = 42
a_string = "MyClass"
another_string = get_string()
reassigned_string = another_string
arr = []
arr[0] = "a string"
arr[1] = MyClass

cyclic_def = my_cyclic_def
my_cyclic_def = cyclic_def

__all__ = [
    MyClass.__name__,
    "MyClass",
    a_string,
    MyStr("foo"),  # OK, inherits from str.
    MyClass,  # Noncompliant
    a, # Noncompliant
    a.b, # FN
    42,  # Noncompliant
    var,  # Noncompliant
    reassigned_var, # Noncompliant
    reassigned_class, # Noncompliant
    # FP here?
    my_global_var, # Noncompliant
    another_string, # OK
    reassigned_string, # OK
    abs, # Noncompliant
    round,  # Noncompliant
    my_func, # Noncompliant
    my_func(), # OK, might return a string
    lambda x: x,  # Noncompliant
    None,  # Noncompliant
    MyClass.my_name, # FN (returns a descriptor)
    MyClass.my_staticmethod, # Noncompliant
    MyClass.my_overloaded_stub, # Noncompliant
    arr[0], # OK
    arr[1], # FN
    "some"  + "string",
    3 + 4, # Noncompliant
    ("some", "string"), # Noncompliant
    a.my_name, # OK
    a.my_staticmethod, # Noncompliant
    a.my_overloaded_stub, # Noncompliant
    unknown_symbol, # FN
    unknown_imported,
    my_cyclic_def
]


def get_string():
  return "some string"
