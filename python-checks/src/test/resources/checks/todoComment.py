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
