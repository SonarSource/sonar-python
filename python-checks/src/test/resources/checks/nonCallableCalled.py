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

    set_var = set()
    set_var()  # Noncompliant

    frozenset_var = frozenset()
    frozenset_var() # Noncompliant

    if p:
      x = 42
    else:
      x = 'str'
    x() # Noncompliant {{Fix this call; "x" is not callable.}}

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
  acos(42)() # Noncompliant {{Fix this call; this expression has type float and it is not callable.}}
# ^^^^^^^^
  wraps(func)(foo) # OK, wraps returns a Callable

def with_metaclass():
  class Factory: ...
  class Base(metaclass=Factory): ...
  class A(Base): ...
  a = A()
  a() # OK

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
