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

    if a is None: # Noncompliant
        pass

def literals():
    a = A()
    if a is 42: # Noncompliant
        pass

    if [] is []: # FN
        pass
