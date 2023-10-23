import math


def math_failure(a):
    math.isclose(a, 0)  # Noncompliant {{Provide the "abs_tol" parameter when using "math.isclose" to compare a value to 0.}}
#   ^^^^^^^^^^^^    ^< {{This argument evaluates to zero.}}
    math.isclose(0, a)  # Noncompliant
#   ^^^^^^^^^^^^ ^<
    b = 0
    math.isclose(a, b)  # Noncompliant
#   ^^^^^^^^^^^^    ^<
    math.isclose(b, a)  # Noncompliant
#   ^^^^^^^^^^^^ ^<

    from math import isclose
    isclose(a, 0)  # Noncompliant {{Provide the "abs_tol" parameter when using "math.isclose" to compare a value to 0.}}
#   ^^^^^^^    ^<

    # this raises an issues as the call is not legitimate ("abs_tol" is a named parameter only)
    math.isclose(0, a, 1, 2)  # Noncompliant
#   ^^^^^^^^^^^^ ^<

    math.isclose(0, 0)  # Noncompliant
#   ^^^^^^^^^^^^ ^<


def isclose(a, b):
    ...


def math_success(a, b):
    math.isclose(42, 4)  # Compliant
    math.isclose(a, 2)  # Compliant
    c = 42
    math.isclose(c, a)  # Compliant
    math.isclose(a, 0, abs_tol=1e-09)  # Compliant
    math.isclose(0, a, abs_tol=1e-09)  # Compliant

    math.isclose(a, b)  # Compliant

    isclose(a, 0)  # Compliant local is close
