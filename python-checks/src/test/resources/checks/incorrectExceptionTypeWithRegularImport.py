import incorrectExceptionTypeImported2
from incorrectExceptionTypeImported3 import SomeException, SomeNotException, SomeChildException, SomeChildNotException, Enclosing
from incorrectExceptionTypeImported4 import RedefinedBaseExceptionChild, ChildOfActualException

def raise_exception():
  raise SomeNotException() # Noncompliant

def raise_exception():
  raise incorrectExceptionTypeImported2.A() # OK

def raise_exception_derived():
  raise incorrectExceptionTypeImported2.DerivedA() # OK

def raise_not_an_exception():
  raise incorrectExceptionTypeImported2.B() # Noncompliant

def raise_not_an_exception_derived():
  raise incorrectExceptionTypeImported2.DerivedB() # Noncompliant

def raise_exception():
  raise SomeException()

def raise_child_exception():
  raise SomeChildException()

def raise_child_exception():
  raise SomeChildNotException() # Noncompliant

def raise_nested_non_exception_class():
  raise Enclsoing.Nested() # FN as only top-level imported symbols are considered

def raise_RedefinedBaseExceptionChild():
  raise RedefinedBaseExceptionChild() # Noncompliant

def raise_ChildOfActualException():
  raise ChildOfActualException() # OK
