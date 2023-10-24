import numpy as np


def failure():
    b = np.bool(True)  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "bool".}}
    #   ^^^^^^^
    i = np.int(42)  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "int".}}
    #   ^^^^^^
    f = np.float(4.2)  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "float".}}
    #   ^^^^^^^^
    c = np.complex(-2.0, 0.0)  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "complex".}}
    #   ^^^^^^^^^^
    o = np.object  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "object".}}
    #   ^^^^^^^^^
    s = np.str("foo")  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "str".}}
    #   ^^^^^^
    l = np.long(123)  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "int".}}
    #   ^^^^^^^
    u = np.unicode("bar")  # Noncompliant {{Replace this deprecated "numpy" type alias with the builtin type "str".}}
    #   ^^^^^^^^^^

def failure_import_as_z():
    import numpy as z
    b = z.bool(True)  # Noncompliant
    #   ^^^^^^
    i = z.int(42)  # Noncompliant
    #   ^^^^^
    f = z.float(4.2)  # Noncompliant
    #   ^^^^^^^
    c = z.complex(-2.0, 0.0)  # Noncompliant
    #   ^^^^^^^^^
    o = z.object  # Noncompliant
    #   ^^^^^^^^
    s = z.str("foo")  # Noncompliant
    #   ^^^^^
    l = z.long(123)  # Noncompliant
    #   ^^^^^^
    u = z.unicode("bar")  # Noncompliant
    #   ^^^^^^^^^

def success():
    b = True  # Compliant
    b = bool(True)  # Compliant
    i = 42  # Compliant
    i = int(42)  # Compliant
    f = 4.2  # Compliant
    f = float(4.2)  # Compliant
    c = complex(-2.0, 0.0)  # Compliant
    o = object  # Compliant
    s = "foo"  # Compliant
    s = str("foo")  # Compliant
    l = 123  # Compliant
    l = int(123)  # Compliant
    u = "bar"  # Compliant
    u = str("bar")  # Compliant

