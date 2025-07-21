def divide(numerator, denominator):
# Noncompliant@+1 {{Is #noqa used to exclude false-positive or to hide real quality flaw?}}
    return numerator / denominator              # noqa denominator value might be 0
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Noncompliant@+1
#noqa

# Noncompliant@+1
    # noqa: E501

# Noncompliant@+1
# noqa: E501,W503

# a noqa just mixed in should not be detected

# NOQA should not be detected 
# NOSONAR should not be detected (different rule)

# NoQa in mixed case should not be detected (only lowercase noqa is valid)

# no qa with space should not be detected

# Invalid example
# noqa:
# Noncompliant@-1 

#  something noqa: something, not valid since not at the beginning of the comment

# this is not no qa
a = "noqa-text"
text = "noqa is mentioned here"
