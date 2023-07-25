foobar = "foobar"

def unknownA():
    ...

def unknownB():
    ...

def returns_string():
    return "foobar"

def f():
  foobar[:3] == 'foo' # Noncompliant {{Use `startsWith` here.}}
# ^^^^^^^^^^^^^^^^^^^
  'foo' == foobar[:3] # Noncompliant

  foobar[3:] == 'bar' # Noncompliant {{Use `endsWith` here.}}
# ^^^^^^^^^^^^^^^^^^^
  'bar' == foobar[3:] # Noncompliant

  if foobar[:3] == 'foo': # Noncompliant
#    ^^^^^^^^^^^^^^^^^^^
    pass

  foobar[:3] != 'foo' # Noncompliant {{Use `not` and `startsWith` here.}}
  'foo' != foobar[:3] # Noncompliant
  foobar[3:] != 'bar' # Noncompliant {{Use `not` and `endsWith` here.}}
  'bar' != foobar[3:] # Noncompliant

  unknownA()[3:] == unknownB() # Compliant but potential FN: The rule should only apply if we are sure that strings are compared
  unknownA()[3:] != unknownB() # Compliant
  unknownA()[3:] == 'bar' # Noncompliant
  unknownA()[:3] == 'foo' # Noncompliant
  unknownA()[:3] != 'foo' # Noncompliant
  'bar' == unknownA()[3:] # Noncompliant
  'foo' == unknownA()[:3] # Noncompliant
  'foo' != unknownA()[:3] # Noncompliant
  'bar'[3:] == unknownA() # Noncompliant
  'foo'[:3] == unknownA() # Noncompliant
  unknownA()[3:] == returns_string() # Noncompliant

  foobar[:3] > 'foo' # Compliant: The rule should not react to any other comparison operators besides == and !=
  'foo' < foobar[:3] # Compliant: The rule should not react to any other comparison operators besides == and !=

  foobar[:3] # Compliant: Slicing outside of a condition
  foobar[3:] # Compliant
  unknownA()[3:] # Compliant

  notastring = [1, 2, 3, 4, 5, 6]
  notastring[:3] == [1, 2, 3] # Compliant: The rule should apply to strings only
  notastring[3:] == [4, 5, 6] # Compliant: The rule should apply to strings only

  # We should definitely also detect the cases with too small / large slices
  foobar[:2] == 'foo' # Noncompliant
  foobar[:10] == 'foo' # Noncompliant
  foobar[2:] == 'bar' # Noncompliant
  foobar[10:] == 'bar' # Noncompliant

  foobar[3:6:1] == 'bar' # Compliant: Too much potential for FPs with step sizes != 1, too early stop indices, etc.
  foobar[-3:] == 'bar' # Noncompliant {{Use `endsWith` here.}}
  foobar[:-3] == 'foo' # Noncompliant {{Use `startsWith` here.}}

  # If an index is not a plain number, we can not say for sure if it is negative/positive. Hence, we should suggest startsWith and endsWith
  foobar[:unknownA()] == 'foo' # Noncompliant {{Use `startsWith` or `endsWith` here}}
  foobar[unknownB():] == 'bar' # Noncompliant {{Use `startsWith` or `endsWith` here}}
  foobar[:-unknownA()] == 'foo' # Noncompliant {{Use `startsWith` or `endsWith` here.}}
  foobar[-unknownA():] == 'bar' # Noncompliant {{Use `endsWith` or `endsWith` here.}}
  foobar[:unknownA()] != 'foo' # Noncompliant {{Use `not` and `startsWith` or `endsWith` here}}

  foobar[3,] == 'foo' # Compliant
  foobar["Hello World"] == 'foo' # Compliant
  s = slice(3)
  foobar[s] == 'foo' # Noncompliant

  class ImplementsIndex:
    def __index__(self):
        return 3

  foobar[ImplementsIndex():] == 'bar' # Noncompliant

  foobar[None:3] == 'foo' # Noncompliant {{Use `startsWith` here.}}
  foobar[None:3:] == 'foo' # Noncompliant {{Use `startsWith` here.}}
  foobar[None:3:None] == 'foo' # Noncompliant {{Use `startsWith` here.}}
  foobar[:3:] == 'foo' # Noncompliant {{Use `startsWith` here.}}
  foobar[3:None] == 'bar' # Noncompliant {{Use `endsWith` here.}}
  foobar[3:None:] == 'bar' # Noncompliant {{Use `endsWith` here.}}
  foobar[3:None:None] == 'bar' # Noncompliant {{Use `endsWith` here.}}
