import tensorflow as tf

class Noncompliant(tf.Module):
  def __init__(self):
    self.count = None

  @tf.function
  def __call__(self):
    self.count = tf.Variable(0) # Noncompliant  {{Refactor this variable declaration to be a singleton.}}
#                ^^^^^^^^^^^^^^
    return self.count.assign_add(1)


class Compliant(tf.Module):
  def __init__(self):
    self.count = None

  @tf.function
  def __call__(self):
    if self.count is None:
      self.count = tf.Variable(0) # OK
    return self.count.assign_add(1)


def non_tf_function():
    x = tf.Variable(42)


@tf.function
def tf_function_outside_class():
    x = tf.Variable(42)  # Noncompliant


@tf.function
def tf_function_constant():
    x = tf.constant(42)  # OK


@tf.function
def tf_function_unknown_call():
    x = unknown()  # OK


if cond:
    @tf.function
    def tf_function_inside_if():
        x = tf.Variable(42)  # Noncompliant


@tf.function
def declared_in_elif():  # OK
    if ...:
        ...
    elif ...:
        x = tf.Variable(42)


class EdgeCases(tf.Module):
    @tf.function
    def conditional_expression(self):  # OK
        self.count = tf.Variable(0) if self.count is None else self.count

    @tf.function
    def while_loop(self):
    # Technically FP as the loop could be designed to execute only once, but we assume it can actually loop
        while self.not_initialized():
            self.x = tf.Variable(42)  # Noncompliant
        return self.x

    @tf.function
    def for_loop(self):
    # Technically FP as the loop could be designed to execute only once, but we assume it can actually loop
        for i in self.initialization_range():
            self.x = tf.Variable(42)  # Noncompliant
        return self.x


x = tf.Variable(42)


class UnrelatedClass:
    @tf.function
    def tf_function_in_unrelated_class():
        x = tf.Variable(42)  # Noncompliant


def tf_different_import():
    import tensorflow as renamed_tf
    @renamed_tf.function
    def tf_function_other_name():
        x = tf.Variable(42)  # Noncompliant


def tf_function_imported_alone():
    from tensorflow import function as renamed_func
    @renamed_func
    def tf_function_other_name():
        x = tf.Variable(42)  # Noncompliant
