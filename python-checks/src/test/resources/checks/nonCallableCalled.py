class MyNonCallable: ...

class MyCallable:
    def __call__(self):
        print("called")

def func(): ...

def call_noncallable():
    myvar = MyNonCallable()
    myvar()  # Noncompliant {{Fix this call; "myvar" is not callable.}}
#   ^^^^^

    none_var = None
    none_var()  # Noncompliant

    int_var = 42
    int_var()  # Noncompliant

    list_var = []
    list_var()  # Noncompliant

    tuple_var = ()
    tuple_var()  # Noncompliant

    dict_var = {}
    dict_var()  # Noncompliant

    set_var = set()
    set_var()  # FN (set has unresolved type hierarchy)

def flow_sensitivity():
  my_var = "hello"
  my_var = 42
  my_var() # Noncompliant

  my_other_var = func
  my_other_var() # OK
  my_other_var = 42
  my_other_var() # FN

def member_access():
  my_callable = MyCallable()
  my_callable.non_callable = 42
  my_callable.non_callable() # FN

def types_from_typeshed():
  from math import acos
  acos(42)() # Noncompliant {{Fix this call; this expression is not callable.}}
# ^^^^^^^^

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
