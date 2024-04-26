class MyNonCallable: ...
#     ^^^^^^^^^^^^^>  {{Definition.}}

class MyCallable:
    def __call__(self):
        print("called")

def func(): ...

def call_noncallable(p):
    myvar = MyNonCallable()
    myvar()  # Noncompliant {{Fix this call; "myvar" has type MyNonCallable and it is not callable.}}
#   ^^^^^

    none_var = None
    none_var()  # Noncompliant {{Fix this call; "none_var" has type NoneType and it is not callable.}}

    int_var = 42
    int_var()  # Noncompliant {{Fix this call; "int_var" has type int and it is not callable.}}

    list_var = []
    list_var()  # Noncompliant

    tuple_var = ()
    tuple_var()  # Noncompliant

    dict_var = {}
    dict_var()  # Noncompliant

    set_literal = {1, 2}
    set_literal()  # Noncompliant

    set_var = set()
    set_var()  # FN

    frozenset_var = frozenset()
    frozenset_var() # FN

    if p:
      x = 42
    else:
      x = 'str'
    x() # FN: multiple assignment not handled

def flow_sensitivity():
  my_var = "hello"
  my_var = 42
  my_var() # FN: multiple assignment not handled

  my_other_var = func
  my_other_var() # OK
  my_other_var = 42
  my_other_var() # FN: multiple assignment not handled

def flow_sensitivity_nested_try_except():
  def func_with_try_except():
    try:
      ...
    except:
      ...

  def other_func():
    my_var = "hello"
    my_var = 42
    my_var() # FN: multiple assignments

def member_access():
  my_callable = MyCallable()
  my_callable.non_callable = 42
  my_callable.non_callable() # FN

def types_from_typeshed(foo):
  from math import acos
  from functools import wraps
  acos(42)() # FN: declared return type of Typeshed
  wraps(func)(foo) # OK, wraps returns a Callable

def with_metaclass():
  class Factory: ...
  class Base(metaclass=Factory): ...
  class A(Base): ...
  a = A()
  # TODO: resolve type hierarchy and metaclasses
  a() # Noncompliant


def decorators():
  x = 42
  @x() # Noncompliant
  def foo():
    ...

#######################################
# Valid case: Calling a callable object
#######################################

def call_callable():
    myvar = MyCallable()
    myvar()

#############################
# Valid case: Call a function
#############################

def call_function():
    from math import max
    func()
    max()

#############################
# Valid case: Call a decorator
#############################

def decorators():
    class Dec:
        def __call__(self, *args):
            ...

    @Dec("foo")
    def foo():
        ...

#############################
# Valid case: Call a method
#############################

class ClassWithMethods:
    def mymethod(self): ...

def call_function():
    myvar = ClassWithMethods()
    myvar.mymethod()  # OK


##########################################################
# Out of scope: detecting that properties cannot be called
#
# A property is not callable, but the value returned
# by the property might be and we are not yet able to know
# if this is the case.
##########################################################

class CustomProperty(property):
    """ test subclasses """

class ClassWithProperties:
    @property
    def prop(self):
        return None

    @prop.setter
    def prop(self, value):
        self._prop = value

    @CustomProperty
    def custom_prop(self):
        return None

    @property
    def callable_prop(self):
        return max

def call_properties():
    myvar = ClassWithProperties()
    myvar.prop()  # FN
    myvar.custom_prop()  # FN
    myvar.callable_prop(1, 2)  # OK


class CalledAtModuleLevel:
  ...

module_level_object = CalledAtModuleLevel()
module_level_object() # Noncompliant


def collections_named_tuple_no_fp():
    from collections import namedtuple
    MyNamedTuple = namedtuple('Employee', ['name', 'age', 'title'])
    x = MyNamedTuple()


def typing_named_tuple_no_fp():
    from typing import NamedTuple
    Employee = NamedTuple('Employee', [('name', str), ('id', int)])
    employee = Employee("Sam", 42)


class Parent:
    def __call__(self):
        ...

class Child(Parent):
    ...


def inherited_call_method():
    child = Child()
    child()  # OK


some_global_func = None

def assigning_global(my_func):
    global some_global_func
    some_global_func = my_func

def calling_global_func():
    some_global_func() # OK


some_nonlocal_var = 42

def using_nonlocal_var():
    nonlocal some_nonlocal_var
    some_nonlocal_var()  # Noncompliant
