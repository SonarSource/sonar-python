def divide(numerator, denominator):
# Noncompliant@+1 {{Is 'noqa' used to exclude false-positive or to hide real quality flaw?}}
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

# --- nosec cases ---
# S1309 is raised for both noqa and nosec suppression directives, so they share this fixture.

# Noncompliant@+1 {{Is 'nosec' used to exclude false-positive or to hide real security issue?}}
import ssl                                      # nosec
#                                               ^^^^^^^

# Noncompliant@+1
import ssl  # nosec B101

# Noncompliant@+1
import ssl  # nosec S4423, S5332 reason text

# Noncompliant@+1
# NOSEC

# a nosec just mixed in should not be detected

# no sec with space should not be detected

# Noncompliant@+1 {{Is 'nosec' used to exclude false-positive or to hide real security issue?}}
# some comment followed by # nosec should be detected
