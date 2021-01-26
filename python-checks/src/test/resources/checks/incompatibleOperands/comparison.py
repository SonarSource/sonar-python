def builtin_noncompliant(p):
  1 < "1"  # Noncompliant {{Fix this invalid "<" operation between incompatible types (int and str).}}
#   ^
  complex(1) < complex(1)  # Noncompliant
  [1] < (1,)  # Noncompliant

  "1" > 1 # Noncompliant {{Fix this invalid ">" operation between incompatible types (str and int).}}

  if p:
    x = 42
  else:
    x = complex(1)
  x > "1" # Noncompliant {{Fix this invalid ">" operation between incompatible types.}}
  "1" < x # Noncompliant


def builtin_compliant():
  1 < int("1")
  4 > 3
  float(4) < 4
  d1 = dict()
  d2 = dict()
  r = d1 | d2


def custom_noncompliant():
  class Empty:
    pass

  class LessThan:
    def __lt__(self, other):
      return True
    def __le__(self, other):
      return True

  empty = Empty()
  lessThan = LessThan()

  empty < 1  # Noncompliant
  empty == 1 # OK
  1 > empty  # Noncompliant
  lessThan < 1  # Ok
  lessThan <= 1  # Ok
  lessThan > 1  # Noncompliant
  lessThan >= 1  # Noncompliant
  1 < lessThan  # Noncompliant
  empty < lessThan  # Noncompliant
  lessThan < empty  # Ok
  empty > lessThan  # Ok

def custom_compliant():
  class A:
    def __lt__(self, other):
      return True

  A() < 1

  class B:
    def __gt__(self, other):
      return True
    def __ge__(self, other):
      return True

  1 < B()
  1 <= B()

def edge_cases():
  class C1:
    def mylt(self, other):
      return True
    __lt__ = mylt

  C1() < 1

  class C2(Unknown):
    pass
  C2() < 1
  1 < C2()

  class C3:
    def __lt__(self, other, p):
      return True

  C3() < 1

def classes_with_decorators():
  @comparable
  class A: pass
  a = A()
  a < 1 # OK
