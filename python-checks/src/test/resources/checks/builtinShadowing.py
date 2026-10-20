def abc():
    max = 42  # Noncompliant {{Rename this variable; it shadows a builtin.}}
   #^^^
    max = foo()
   #^^^<

    int : int = 42  # Noncompliant {{Rename this variable; it shadows a builtin.}}
   #^^^

def bcd():
    max = 42  # Noncompliant {{Rename this variable; it shadows a builtin.}}
   #^^^

    foo(x=(int := f(x)))  # Noncompliant {{Rename this variable; it shadows a builtin.}}
    #      ^^^

max = 42
foo(x=(max := f(x)))
int : int
safe : int = 42
42 : int = 42

class efg():
    int : int
    max = 42

def safe():
    abs : int

def redefine():
    global __doc__
    __doc__ = 42

def iPython():
    display = 42

def ellipsis_test():
    ellipsis = "" # OK
    Ellipsis = 42 # Noncompliant {{Rename this variable; it shadows a builtin.}}
    ellipsis = "..." # OK

def function_test():
    function = "" # OK


def global_builtin():
    global int
    int = 42


def nested_bindings():
    len = 1  # Noncompliant
    def inner():
        len = 2  # Noncompliant
        return len
    return len


def nonlocal_binding():
    sum = 1  # Noncompliant
   #^^^
    def inner():
        nonlocal sum
        sum = 2
       #^^^<
        return sum
    return sum


def imported_module_is_not_a_builtin():
    import os
    os = None


def nested_global_does_not_hide_local():
    int = 1  # Noncompliant
   #^^^
    def inner():
        global int
        int = 2
    int = 3
   #^^^<
    return int


def nested_class_global_does_not_hide_local():
    int = 1  # Noncompliant
    class Nested:
        global int
        int = 2
    return int
