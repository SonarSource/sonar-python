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
