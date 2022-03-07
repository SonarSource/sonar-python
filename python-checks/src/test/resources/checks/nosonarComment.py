def divide(numerator, denominator):
# Noncompliant@+1
    return numerator / denominator              # NOSONAR denominator value might be 0
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Noncompliant@+1
#NOSONAR

# Noncompliant@+1
# this is NOSONAR

# nosonar in lower case

# NoSonAr in mixed case

for d in lib_dirs:
    # NOSONAR:
    # Noncompliant@-1
    pass

if True:
    print("a")
    # Noncompliant@+1
    #  something NOSONAR: something
    for d in lib_dirs:
        pass
# this is not no sonar
a = "NOSONAR-text"
