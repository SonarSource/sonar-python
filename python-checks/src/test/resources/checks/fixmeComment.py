def divide(numerator, denominator):
# Noncompliant@+1 {{Take the required action to fix the issue indicated by this "FIXME" comment.}}
    return numerator / denominator              # FIXME denominator value might be 0
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Noncompliant@+1
#FIXME

# this is not fixme

# Noncompliant@+1
# fixme in lower case

for d in lib_dirs:
    # FIXME: some fixme
    # Noncompliant@-1
    # not a fix me
    pass

if True:
    print("a")
    # Noncompliant@+1
    # FIXME: something
    for d in lib_dirs:
        pass
# fix me and this is not
