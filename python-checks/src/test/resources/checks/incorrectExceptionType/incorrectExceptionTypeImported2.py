from incorrectExceptionTypeImported3 import Animal

class A(BaseException):
  pass

class B():
  pass

class DerivedB(B):
  pass

class DerivedA(A):
  pass

def func():
  pass

class Dog(Animal):
    pass
