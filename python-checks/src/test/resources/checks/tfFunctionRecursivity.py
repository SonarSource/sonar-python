import tensorflow as tf

@tf.function
def factorial(n):
    if n == 1:
        return 1
    else:
        return (n * factorial(n-1)) # Noncompliant {{Remove this recursive call.}}
                   #^^^^^^^^^^^^^^
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
