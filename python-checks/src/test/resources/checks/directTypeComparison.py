class A:
    ...

def foo(a, b):
    if type(a) == str:  # Noncompliant {{Use the `isinstance()` function here.}}
#              ^^
        ...
    if type(a) == A: # Noncompliant
        ...
    if isinstance(a, str):
        ...
    if str == type(a):  # Noncompliant
        ...
    if bar(a) == str:
        ...
    if unknown(a) == str:
        ...
    t = type(a)
    if t == str: # FN
        ...
    if type(a) == type(b): # OK
        ...

    if type(a) == get_class(): # FN
        ...

    if type(a) > str: # OK, not relevant for this rule
        ...

    if type(a) != str: # Noncompliant {{Use `not isinstance()` here.}}
        ...

def get_class():
    return str


def bar(a):
    ...
