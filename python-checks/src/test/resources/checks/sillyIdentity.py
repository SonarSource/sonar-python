class A:
    pass

class B:
    pass

def foo():
    a = A()
    b = B()
    if a is b: # Noncompliant {{Remove this "is" check; it will always be False.}}
    #    ^^
        pass
    if a is not b: # Noncompliant {{Remove this "is not" check; it will always be True.}}
    #    ^^^^^^
        pass

    a2 = A()
    if a is a2: # OK
        pass

def numeric_literals():
    a = A()
    if a is 42: # Noncompliant
        pass

    if 42.0 is 42: # Noncompliant
        pass

    if 42 is 42: # OK
        pass

    if 42.0 is 42.0: # OK
        pass


def string_literals():
    a = A()
    if a is "foo": # Noncompliant
        pass

    if b"foo" is "foo": # Noncompliant
        pass

    if "foo" is "foo": # OK
        pass

    if 42 is "42": # Noncompliant
        pass

def other_literals():
    a = A()
    if a is []: # Noncompliant
        pass

    if [] is []: # FN
        pass

    if [] is {}: # Noncompliant
        pass
