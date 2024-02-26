from some_module import unknown
from typing import Sequence, Iterable, Container, Callable, overload

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
#  ^^ ^^^^< {{The result value of this expression does not support the membership protocol.}}

42 not in 1337 # Noncompliant
#  ^^^^^^ ^^^^< {{The result value of this expression does not support the membership protocol.}}

if 42 in 1337: # Noncompliant
#     ^^ ^^^^<
    pass

42 in int(this_is() + a_long() + # Noncompliant {{Change the type for the target expression of `in`; type int does not support membership protocol.}}
#  ^^ ^[el=+2;ec=29]<
      multiline_expression())

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
    # TODO: Consider adding this rule to ConfusingTypeChecking, see the example below
    x in produces_empty() # Compliant since type analysis makes no assumptions about the members of declared types
    "a" in "abc" # Compliant
    b'a' in b'abc' # Compliant

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

    class DisabledContainsWithIter:
        __contains__ = None
        def __iter__(self):
            ...

    x in DisabledContainsWithIter() # Noncompliant

    class ContainsWithDisabledIter:
        def __contains__(self, item):
            ...

        __iter__ = None

    x in ContainsWithDisabledIter() # Compliant

    class DisabledIterWithGetItem:
        __iter__ = None

        def __getitem__(self, item):
            ...


    x in DisabledIterWithGetItem() # Noncompliant

    class IterWithDisabledGetItem:
        def __iter__(self):
            ...

        __getitem__ = None

    x in IterWithDisabledGetItem() # Compliant

    class AmbiguousContains01:
        __contains__ = unknown()

    x in AmbiguousContains01() # Compliant

    class AmbiguousContains02:
        __contains__ = unknown()
        __iter__ = None

    x in AmbiguousContains02() # Compliant

    class AmbiguousIter:
        __iter__ = unknown()

    x in AmbiguousIter() # Compliant

    class AmbiguousIterAndDisabledContains:
        __contains__ = None
        __iter__ = unknown()

    x in AmbiguousIterAndDisabledContains() # Noncompliant

    class AmbiguousContainsAndIter:
        __contains__ = unknown()
        def __iter__(self):
            ...

    x in AmbiguousContainsAndIter() # Compliant

    class ComplexContains01:
        __contains__ = None
        __contains__ = unknown()

    x in ComplexContains01() # Compliant

    class ComplexContains02:
        __contains__: Callable

    x in ComplexContains02() # Compliant

    class ComplexContains03(ComplexContains01, ComplexContains02):
        pass

    x in ComplexContains03() # Compliant

    class ClassSymbolContains:
        class __contains__: # Ridiculous case (and so are the others here), but I add it for coverage
            pass

    x in ClassSymbolContains() # Noncompliant

    class OverloadedContains:
        @overload
        def __contains__(self, item: int) -> bool:
            ...
        @overload
        def __contains__(self, item: float) -> bool:
            ...
        def __contains__(self, item):
            ...

    x in OverloadedContains() # Compliant

def no_fp_on_enum_types():
    from enum import Enum

    class EnumA(Enum):
        A = 1
        B = 2
        C = 3

    selected = foo()
    selected in EnumA # OK
