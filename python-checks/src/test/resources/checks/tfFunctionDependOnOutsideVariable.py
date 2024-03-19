import tensorflow as tf

foo = 3

@tf.function
def non_compliant(): # Noncompliant {{This function should not depend implicitly on a global or free variable.}}
   #^^^^^^^^^^^^^
    return foo + 1
          #^^^< {{Variable used here.}}

no_symbol = True
def no_symbol(): ...
@tf.function
def non_compliant2(): # Noncompliant {{This function should not depend implicitly on a global or free variable.}}
    something_else = no_symbol
    return something_else + 1

@tf.function
def non_compliant3(): # Noncompliant
   #^^^^^^^^^^^^^^
    something = foo
               #^^^< {{Variable used here.}}
    smth = no_symbol
          #^^^^^^^^^< {{Variable used here.}}
    return foo + no_symbol
          #^^^< {{Variable used here.}}
                #^^^^^^^^^@-1< {{Variable used here.}}

@tf.function
def non_compliant4(foo): # Noncompliant
   #^^^^^^^^^^^^^^
    something = foo
    smth = no_symbol
          #^^^^^^^^^< {{Variable used here.}}
    return foo + no_symbol
                #^^^^^^^^^< {{Variable used here.}}
@tf.function
def compliant(foo):
    return foo + 1

from somewhere import something

class SomeClass:
    ...

@tf.function
def other_kind_of_name(foo):
    def bar():
        ...
    some_var = bar
    some_other_var = bar()
    some = something()
    some_object = SomeClass()
    return bar(foo)

def some_other_function():
    return 42

def indirect_tf_function():
    return foo + 1 # FN : we don't recursively check if we are a tf.function

@tf.function
def direct_tf_function():
    return indirect_tf_function()
