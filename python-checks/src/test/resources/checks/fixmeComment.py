def divide(numerator, denominator):
# Noncompliant@+1 {{Take the required action to fix the issue indicated by this "FIXME" comment.}}
    return numerator / denominator              # FIXME denominator value might be 0
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Noncompliant@+1
#FIXME

# this is not fixme

# Noncompliant@+1
# fixme in lower case

# fix me and this is not
