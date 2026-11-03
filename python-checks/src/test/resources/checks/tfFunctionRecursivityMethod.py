import tensorflow as tf


class RecursiveMethod:
    # fmt: off
    @tf.function
    def recurse(self, n): # Noncompliant {{Make sure to avoid recursive calls in this function.}}
       #^^^^^^^
        return self.recurse(n - 1)
        #      ^^^^^^^^^^^^< {{Recursive call is here.}}
    # fmt: on


class NonRecursiveMethod:
    @tf.function
    def recurse(self, n):
        return self.other(n)

    def other(self, n):
        return n
