def divide(numerator, denominator):
# Noncompliant@+1 {{Complete the task associated to this "TODO" comment.}}
    return numerator / denominator              # TODO denominator value might be 0
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Noncompliant@+1
#TODO

# this is not TODO

# Noncompliant@+1
# todo in lower case

for d in lib_dirs:
    # TODO: some TODO
    # Noncompliant@-1
    pass

if True:
    print("a")
    # Noncompliant@+1
    # TODO: something
    for d in lib_dirs:
        pass

# TODO_something
#      TODO132

# todos all in spanish
# ToDo app

# Noncompliant@+1
#      TODO more spaces

# Noncompliant@+1
# Todo first capital

# Noncompliant@+1
# Todo-dash

# Noncompliant@+1
# TODO(FIX: parenthesis)


