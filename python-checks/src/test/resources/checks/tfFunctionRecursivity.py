import tensorflow as tf

@tf.function
def factorial(n): # Noncompliant {{Make sure to avoid recursive calls in this function.}}
   #^^^^^^^^^
    if n == 1:
        return 1
    else:
        return (n * factorial(n-1))
                   #^^^^^^^^^< {{Recursive call is here.}}

@tf.function
def multiple_recursive_calls(n): # Noncompliant {{Make sure to avoid recursive calls in this function.}}
   #^^^^^^^^^^^^^^^^^^^^^^^^
    if cond:
        multiple_recursive_calls(n-1)
       #^^^^^^^^^^^^^^^^^^^^^^^^< {{Recursive call is here.}}
    else:
        multiple_recursive_calls(n-1)
       #^^^^^^^^^^^^^^^^^^^^^^^^< {{Recursive call is here.}}
    return multiple_recursive_calls(n+1) + multiple_recursive_calls(n-1)
          #^^^^^^^^^^^^^^^^^^^^^^^^< {{Recursive call is here.}}
                                          #^^^^^^^^^^^^^^^^^^^^^^^^@-1< {{Recursive call is here.}}

@tf.function
def indirect_rec1(n): # FN because of indirect recursion
    return indirect_rec2(n)
@tf.function
def indirect_rec2(n): # FN because of indirect recursion
    return indirect_rec1(n)

@tf.function
def unknown_call(n):
    return some_function(n)

@tf.function
def symbol_not_function(x):
    indirect_call = symbol_not_function
    return indirect_call(x) # FN

@tf.function
def other_function(n):
    return n
@tf.function
def call_another_function():
    return other_function(42)

def compliant_factorial(n):
    if n == 1:
        return 1
    else:
        return (n * compliant_factorial(n-1))

factorial(5)

@tf.function
def ambiguous(n):
    return ambiguous(n-1) # FN because of ambiguous symbol

@tf.function
def ambiguous(n, n1):
    return ambiguous(n-1, n1) # FN because of ambiguous symbol

def other_names():
    import tensorflow as not_tf
    @not_tf.function
    def not_tf_function(n): # Noncompliant {{Make sure to avoid recursive calls in this function.}}
        return not_tf_function(n-1)

    from tensorflow import function as our_decorator
    @our_decorator
    def our_decorated_function(n): # Noncompliant {{Make sure to avoid recursive calls in this function.}}
        return our_decorated_function(n-1)

    @unknown_decorator
    def some_function(): ...