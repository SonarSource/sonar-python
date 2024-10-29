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
    set_var()  # Noncompliant

    frozenset_var = frozenset()
    frozenset_var() # Noncompliant

    if p:
      x = 42
    else:
      x = 'str'
    x() # Noncompliant


def call_no_name():
    42() # Noncompliant {{Fix this call; this expression has type int and it is not callable.}}

def flow_sensitivity():
  my_var = "hello"
  my_var = 42
  my_var() # Noncompliant

  my_other_var = func
  my_other_var() # OK
  my_other_var = 42
  my_other_var() # Noncompliant

def flow_sensitivity_nested_try_except():
  def func_with_try_except():
    try:
      ...
    except:
      ...

  def other_func():
    my_var = "hello"
    my_var = 42
    my_var() # Noncompliant

def member_access():
  my_callable = MyCallable()
  my_callable.non_callable = 42
  my_callable.non_callable() # FN

def types_from_typeshed(foo):
  from math import acos
  from functools import wraps
  acos(42)() # Noncompliant
  wraps(func)(foo) # OK, wraps returns a Callable

def with_metaclass():
  class Factory: ...
  class Base(metaclass=Factory): ...
  class A(Base): ...
  a = A()
  a()


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
    some_global_func()  # OK


some_nonlocal_var = 42

def using_nonlocal_var():
    nonlocal some_nonlocal_var
    some_nonlocal_var()  # OK


def reassigned_function():
    if cond:
        def my_callable(): ...
        my_callable()  # OK
    else:
        def my_callable(): ...
        my_callable = 42
        my_callable()  # Noncompliant



def recursive_with_try_finally(x):
    if x is False:
        print("recursion!")
        return
    recursive_with_try_finally(False) # Noncompliant
    try:
        recursive_with_try_finally(False) # Noncompliant
    finally:
        recursive_with_try_finally = None
        recursive_with_try_finally(False) # Noncompliant
    recursive_with_try_finally(False)  # Noncompliant



def nested_recursive_try_finally():
    def my_rec(x):
        if x is False:
            print("yeah")
            return
        my_rec(False)
    try:
        my_rec(True)
    finally:
        my_rec = None

def call_non_callable_property():
    e = OSError()
    e.errno()  # Noncompliant

class MyClass:
    x = 42
    my_classmethod = classmethod(...)

def foo():
    mc = MyClass()
    mc.x() # FN
    mc.my_classmethod() # OK

def using_isinstance_with_runtime_type():
    my_non_callable = MyNonCallable()
    if isinstance(my_non_callable, whatever):
        my_non_callable() # Noncompliant
    ...
    my_non_callable()  # Noncompliant

def reassigned_param(a, param):
    param = 1
    if a:
        param = [1,2,3]
    param() # Noncompliant

def conditionaly_reassigned_param(a, param):
    if a:
        param = [1,2,3]
    param()

def reassigned_param_try_except(a, param):
    try:
        param = 1
        if a:
            param = [1,2,3]
        param() # FN
    except:
        ...

def conditionally_reassigned_param_try_except(a, param):
    try:
        if a:
            param = [1,2,3]
        param()
    except:
        ...


def nested_function_in_try_catch():
    foo = None
    try:
        ...
    except:
        ...
    def bar():
        foo()



def f1():
    ...

def f2():
    ...

def callable_from_loop_try_except():
    l = [f1, f2]
    try:
        for i in l:
            i()
    except:
        ...

def non_callable_from_loop_try_except():
    l = ["f1", "f2"]
    try:
        for i in l:
            i() # Noncompliant
    except:
        ...


def callable_from_loop_append_noncallable():
    l = [f1, f2]
    l.append("1")
    for i in l:
        i() # FN


def callable_from_loop_append_noncallable():
    l = ["1"]
    possible_modiffication(l)
    for i in l:
        # FP
        i() # Noncompliant



def add_call_method(cls):
    def new_call_method(self, *args, **kwargs):
        ...
    cls.__call__ = new_call_method
    return cls

@add_call_method
class DecoratedClass:
    ...

@unknown_decorator
class ClassWithUnknownDecorator:
    ...

from dataclasses import dataclass

@dataclass
class Employee:
    name: str
    age: int
    position: str

def no_fp_on_decorated_classes():
    decorated_class = DecoratedClass()
    decorated_class()  # OK
    unknown_decorated_class = ClassWithUnknownDecorator()
    unknown_decorated_class()  # OK
    employee = Employee("John Doe", 30, "Software Engineer")
    employee() # FN


def typing_extensions_typed_dict():
    from typing_extensions import TypedDict
    # TypedDict is defined as "TypedDict: object" in typing_extensions.pyi
    # Despite actually being a function
    x = TypedDict('x', {'a': int, 'b': str}) # OK


def typing_typed_dict():
    from typing_extensions import TypedDict
    # TypedDict is defined as "TypedDict: object" in typing.pyi
    # Despite actually being a function
    x = TypedDict('x', {'a': int, 'b': str}) # OK

def function_type_is_callable():
    import unittest
    # unittest.skip() returns a Callable
    unittest.skip("reason")() # OK


def object_typevar():
    scheduled = []
    scheduled.pop()()  # OK
