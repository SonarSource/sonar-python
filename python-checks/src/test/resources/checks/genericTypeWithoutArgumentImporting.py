from genericTypeWithoutArgumentImported import (
    SomeGeneric,
    SomeGenericWithTypeParam,
    MyImportedGenericTypeVarChild,
    MyImportedNonGenericChild,
    MyImportedConcreteChild,
    SomeGenericIncorrectlyDefined
)

def local_generic():
    from typing import Generic
    class LocalGeneric(Generic[T]):
      ...
    def bar() -> LocalGeneric: # Noncompliant
      ...

def foo() -> SomeGeneric: # Noncompliant
  ...

def bar() -> SomeGenericWithTypeParam: # Noncompliant
    ...

def returning_imported_child() -> MyImportedGenericTypeVarChild: ... # Noncompliant
def returning_imported_non_generic_child() -> MyImportedNonGenericChild: ... # OK
def returning_imported_concrete_child() -> MyImportedConcreteChild: ... # OK

class MyChild(SomeGeneric[T]): ...
def returning_my_child() -> MyChild: # FN
    ...

class MyNonGenericChild(SomeGeneric): ...
def returning_my_non_generic_child() -> MyNonGenericChild: # OK
    ...

class MyConcreteChild(SomeGeneric[str]): ...
def returning_my_concrete_chil3() -> MyConcreteChild: # OK
    ...

def returning_incorrect_generic() -> SomeGenericIncorrectlyDefined:
    ...
