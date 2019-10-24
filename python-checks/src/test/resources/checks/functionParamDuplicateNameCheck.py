def other(a, a, c): # Noncompliant {{Rename the duplicated function parameter "a".}}
#   ^^^^^
    return a * c

def fun1(a, c): # compliant
    [...]
    return a * c

def other2(a, a, c, c, e): # Noncompliant {{Rename the duplicated function parameters "a, c".}}
#   ^^^^^^
    return a * c

def fun2(*):
  pass

def fun3(*a, a): # Noncompliant
  pass
def fun4(a, **a): # Noncompliant
  pass
def fun5(*a, b):
  pass
def fun6(a, **b):
  pass
def foo():
  pass
