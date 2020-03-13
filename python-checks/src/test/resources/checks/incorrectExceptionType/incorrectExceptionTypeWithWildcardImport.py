from incorrectExceptionTypeImported import *

def raise_a_custom_exception():
  raise ACustomException()

def raise_some_class():
  raise SomeClass() # Noncompliant

def raise_some_class_type_inference():
  a = SomeClass()
  raise a # Noncompliant

def raise_a_derived_class():
  raise SomeDerivedClass() # Noncompliant

def raise_a_derived_class():
  a = SomeDerivedClass
  raise a # FN

def raise_a_derived_class_from_unknown():
  raise DerivedClassFromUnknown() # OK

def raise_enclosing_class():
  raise Enclosing() # Noncompliant

def raise_a_nested_class_derived_from_BaseException():
  raise Enclosing.Nested() # OK

def raise_a_nested_non_exception_class():
  raise Enclosing.Nested2() # Noncompliant

def raise_a_nested_class_derived_from_python2_Exception():
  raise DerivedFromPython2Exception() # OK
