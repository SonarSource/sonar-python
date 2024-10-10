
def unknownA():
    ...

def unknownB():
    ...

def returns_string() -> str:
    ...

def f():
  foobar = "foobar"

  foobar[:3] == 'foo' # Noncompliant {{Use `startswith` here.}}
# ^^^^^^^^^^^^^^^^^^^
  'foo' == foobar[:3] # Noncompliant

  foobar[3:] == 'bar' # Noncompliant {{Use `endswith` here.}}
# ^^^^^^^^^^^^^^^^^^^
  'bar' == foobar[3:] # Noncompliant

  if foobar[:3] == 'foo': # Noncompliant
#    ^^^^^^^^^^^^^^^^^^^
    pass

  foobar[:3] != 'foo' # Noncompliant {{Use `not` and `startswith` here.}}
  'foo' != foobar[:3] # Noncompliant
  foobar[3:] != 'bar' # Noncompliant {{Use `not` and `endswith` here.}}
  'bar' != foobar[3:] # Noncompliant

  unknownA()[3:] == unknownB() # Compliant but potential FN: The rule should only apply if we are sure that strings are being compared
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
  foobar[3:6:0x1] == 'bar' # Same
  foobar[-3:] == 'bar' # Noncompliant {{Use `endswith` here.}}
  foobar[:-3] == 'foo' # Noncompliant {{Use `startswith` here.}}

  # If an index is not a plain number, we can not say for sure if it is negative/positive. Hence, we should suggest startswith and endswith
  foobar[:unknownA()] == 'foo' # Noncompliant {{Use `startswith` here.}}
  foobar[unknownB():] == 'bar' # Noncompliant {{Use `endswith` here.}}
  foobar[:-unknownA()] == 'foo' # Noncompliant {{Use `startswith` here.}}
  foobar[-unknownA():] == 'bar' # Noncompliant {{Use `endswith` here.}}
  foobar[:unknownA()] != 'foo' # Noncompliant {{Use `not` and `startswith` here.}}

  foobar[3,] == 'foo' # Compliant: We do not deal with invalid indices. These are handled by S6663
  foobar["Hello World"] == 'foo' # Same
  foobar[:3,3:] == 'foo' # Same
  s = slice(3)
  foobar[s] == 'foo' # Compliant but FN: We do not analyze slice objects
  foobar[slice(3)] == 'foo' # Same

  class ImplementsIndex:
    def __index__(self):
        return 3

  foobar[ImplementsIndex():] == 'bar' # Noncompliant
  foobar[3] == 'foo' # OK: Simple integer indexing, not a slice expression

  foobar[None:3] == 'foo' # Noncompliant {{Use `startswith` here.}}
  foobar[None:3:] == 'foo' # Noncompliant {{Use `startswith` here.}}
  foobar[None:3:None] == 'foo' # Noncompliant {{Use `startswith` here.}}
  foobar[None:3:1] == 'foo' # Noncompliant {{Use `startswith` here.}}
  foobar[:3:] == 'foo' # Noncompliant {{Use `startswith` here.}}
  foobar[3:None] == 'bar' # Noncompliant {{Use `endswith` here.}}
  foobar[3:None:] == 'bar' # Noncompliant {{Use `endswith` here.}}
  foobar[3:None:None] == 'bar' # Noncompliant {{Use `endswith` here.}}
  foobar[None:None:None] == 'bar' # OK: This slice returns the full string, hence comparison by == is probably appropriate. Also, something like this will probably not appear in real code

  foobar[:3:2] == 'fo' # Compliant: This is not a simple step 1 slice, not a prefix, or suffix, so the rule does not apply
  foobar[:3:-2] == 'r' # Same

