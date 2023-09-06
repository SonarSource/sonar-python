import math


def math_failure(a):
    math.isclose(a, 0)  # Noncompliant {{Provide the abs_tol parameter when using math.isclose to compare a value to 0}}
#   ^^^^^^^^^^^^^^^^^^
    math.isclose(0, a)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^
    b = 0
    math.isclose(a, b)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^
    math.isclose(b, a)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^

    from math import isclose
    isclose(a, 0)  # Noncompliant {{Provide the abs_tol parameter when using math.isclose to compare a value to 0}}
#   ^^^^^^^^^^^^^


def isclose(a, b):
    ...


def math_success(a, b):
    math.isclose(a, 2)  # Compliant
    c = 42
    math.isclose(c, a)  # Compliant
    math.isclose(a, 0, abs_tol=1e-09)  # Compliant
    math.isclose(0, a, abs_tol=1e-09)  # Compliant

    math.isclose(a, b)  # Compliant

    isclose(a, 0)  # Compliant local is close
