import tensorflow as tf

@tf.function
def using_print_statement():  # Noncompliant {{Make sure this Tensorflow function doesn't have Python side effects.}}
#   ^^^^^^^^^^^^^^^^^^^^^
    print("hello")
#   ^^^^^< {{Statement with side effect.}}

@tf.function
def using_open_statement():  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    with open("something.txt"): ...
#        ^^^^<


@tf.function
def defining_global_variables():  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^
    global my_global
#   ^^^^^^^^^^^^^^^^<
    ...

@tf.function
def defining_nonlocal_variables():  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    nonlocal my_nonlocal
#   ^^^^^^^^^^^^^^^^^^^^<
    ...

@tf.function
def iterating_over_python_range():  # OK, even though tf.range should be used instead
    for i in range(10):
        ...


@tf.function
def using_collections():  # Noncompliant
#   ^^^^^^^^^^^^^^^^^
    my_list = [1, 2, 3]
    my_list.append(3)
#   ^^^^^^^^^^^^^^<
    my_set = {1, 2, 3}
    my_set.add(4)
#   ^^^^^^^^^^<
    my_dict = {"1": 1, "2": 2}
    my_dict.clear()
#   ^^^^^^^^^^^^^<


@tf.function
def using_del_statements(some_param):  # Noncompliant
    del some_param

@tf.function
def subscription_modifying_list(my_list):  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Because Tensorflow objects cannot be modified in such way, we can assume my_list is a regular Python object that gets mutated
    my_list[4] = 42
#   ^^^^^^^^^^^^^^^<


@tf.function
def using_comprehensions():
    x = [i for i in range(10)]
    y = {i for i in range(10)}

@tf.function
def appending_to_list(my_list):  # FN
    my_list.append("42")


@tf.function
def appending_to_list(my_list: list[str]):  # FN
    my_list.append("42")


def some_normal_function(my_list):
    global some_global
    some_global = 42
    my_list.append("42")
    my_list[4] = 42
    my_other_list = [1, 2, 3]
    my_other_list.append(3)
    print("Hello")


@tf.something_else
def some_unrelated_tf_decorator(my_list):
    global some_global
    some_global = 42
    my_list.append("42")
    my_list[4] = 42
    my_other_list = [1, 2, 3]
    my_other_list.append(3)
    print("Hello")


@some_unknown_decorator
def some_unrelated_tf_decorator(my_list):
    global some_global
    ...


@subscription_decorator[42]
def some_unrelated_tf_decorator(my_list):
    global some_global
    ...
