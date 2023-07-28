from some_module import unknown
from typing import Sequence, Iterable, Container

def produces_sequence() -> Sequence:
    ...

def produces_iterable() -> Iterable:
    ...

def produces_container() -> Container:
    ...

class Empty:
    pass
def produces_empty() -> Empty:
    ...

42 in 1337 # Noncompliant {{Change the type of 1337; type int does not support membership protocol.}}
#  ^^ ^^^^< {{This result of this expression does not support the membership protocol.}}

42 not in 1337 # Noncompliant
#  ^^^^^^ ^^^^< {{This result of this expression does not support the membership protocol.}}

if 42 in 1337: # Noncompliant
#     ^^ ^^^^<
    pass

def f():
    x = 1337
    42 in x # Noncompliant
    y = 42
    y in 1337 # Noncompliant
    x in y # Noncompliant

    x in unknown() # Compliant
    unknown() in x # Noncompliant
    unknown() in unknown() # Compliant

x in [] # Compliant
x in () # Compliant
x in {} # Compliant
x in {v for v in "abc"} # Compliant
x in frozenset()
x in dict() # Compliant
x in {k: 42 for k in "abc"} # Compliant
x in produces_container() # Compliant
x in produces_iterable() # Compliant
x in produces_sequence() # Compliant
x in produces_empty() # Noncompliant
"a" in "abc" # Compliant
b'a' in b'abc' # Compliant

def g():
    list_or_int = []
    if unknown():
        list_or_int = 42

    unknown() in list_or_int # Compliant: At least one of the members of this union type supports the membership protocol

    bool_or_int = True
    if unknown():
        bool_or_int = 42
    # None of the types in this union type support the membership protocol
    unknown() in bool_or_int # Noncompliant

x in Empty() # Noncompliant

class HasContains:
    def __contains__(self, item):
        ...

x in HasContains() # Compliant

class HasIter:
    def __iter__(self):
        ...

x in HasIter() # Compliant


class HasGetItem:
    def __getitem__(self, item):
        ...

x in HasGetItem() # Compliant

class InheritedContains(HasContains):
    pass

x in InheritedContains() # Compliant

class InheritedEmpty(Empty):
    pass

x in InheritedEmpty() # Noncompliant

class Mix011(HasContains, HasIter):
    pass

x in Mix011() # Compliant

class Mix101(HasContains, HasGetItem):
    pass

x in Mix101() # Compliant

class Mix110(HasIter, HasGetItem):
    pass

x in Mix110() # Compliant

class Mix111(HasContains, HasIter, HasGetItem):
    pass

x in Mix111() # Compliant

class InheritedSet(set):
    pass

x in InheritedSet("abc") # Compliant

class DisabledContains(HasContains):
    __contains__ = None

x in DisabledContains() # Noncompliant

