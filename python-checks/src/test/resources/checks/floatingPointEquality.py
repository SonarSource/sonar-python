
def failure(a, b):
    if a == b - 0.1:  # Noncompliant {{Do not perform equality checks with floating point values.}}
#      ^^^^^^^^^^^^
        ...

    if a * .2 == b:  # Noncompliant {{Do not perform equality checks with floating point values.}}
#      ^^^^^^^^^^^
        ...

    if a == b / 0.35:  # Noncompliant {{Do not perform equality checks with floating point values.}}
#      ^^^^^^^^^^^^^
        ...

    if a - .2 == b:  # Noncompliant {{Do not perform equality checks with floating point values.}}
#      ^^^^^^^^^^^
        ...

    if a == 0.1:  # Noncompliant {{Do not perform equality checks with floating point values.}}
#      ^^^^^^^^
        ...

    if 0.1 != b:  # Noncompliant {{Do not perform equality checks with floating point values.}}
#      ^^^^^^^^
        ...

    if a != b - 0.1:  # Noncompliant
#      ^^^^^^^^^^^^
        ...

    c = 3.2
    if a == c:  # Noncompliant
#      ^^^^^^
        ...

    m = 23.4
    if b > 2:
        m = 2.4
    else:
        m = 4.2

    if c == m:  # Noncompliant
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

    m = 23.4
    if b > 2:
        m = 4
    else:
        m = 4.2

    if c == m:  # Compliant
        ...

    if c > 0.2:  # Compliant
        ...
