
def failure(a, b):
    if a == b - 0.1:  # Noncompliant {{Equality tests should not be made with floating point values.}}
#      ^^^^^^^^^^^^
        ...

    if a * .2 == b:  # Noncompliant {{Equality tests should not be made with floating point values.}}
#      ^^^^^^^^^^^
        ...

    if a == b / 0.35:  # Noncompliant {{Equality tests should not be made with floating point values.}}
#      ^^^^^^^^^^^^^
        ...

    if a - .2 == b:  # Noncompliant {{Equality tests should not be made with floating point values.}}
#      ^^^^^^^^^^^
        ...

    if a == 0.1:  # Noncompliant {{Equality tests should not be made with floating point values.}}
#      ^^^^^^^^
        ...

    if 0.1 != b:  # Noncompliant {{Equality tests should not be made with floating point values.}}
#      ^^^^^^^^
        ...

    if a != b - 0.1:  # Noncompliant
#      ^^^^^^^^^^^^
        ...

    c = 3.2
    if a == c:  # Noncompliant
#      ^^^^^^
        ...

    if c == b:  # Noncompliant
#      ^^^^^^
        ...


def success(a, b):
    if a == b:  # Compliant
        ...

    if 2 == b:  # Compliant
        ...

    if a == 1 - 2:  # Compliant
        ...

    c = 123
    if a != c:  # Compliant
        ...

    if c == b:  # Compliant
        ...

    if c > 0.2:  # Compliant
        ...
