from external import unknown
from abc import ABC, abstractmethod
import abc
from some_module import some_decorator
from itertools import chain
from typing import Protocol, Any
import typing
from zope.interface import Interface
import zope

def string_unknown() -> str:
    ...

class IterMethodCheck01:
    def __iter__(self):
        return True # Noncompliant {{Return an object complying with iterator protocol.}}
        #      ^^^^

class IterMethodCheck02:
    def __iter__(self):
        return chain([]) # Compliant

class IterMethodCheck03:
    def __iter__(self):
        x = True
        return x # Noncompliant

class IterMethodCheck04:
    def __iter__(self):
        return unknown() # Compliant but potential for FNs

class IterMethodCheck05:
    def __iter__(self):
        # Strings are iterables that have the __iter__ method, but they lack the __next__method
        return "Hello World" # Noncompliant

class HasIterButNoNext:
    def __iter__(self): # Noncompliant
        ...

class IterMethodCheck06:
    def __iter__(self):
        return HasIterButNoNext() # Noncompliant

class HasNextButNoIter:
    def __next__(self):
        ...

class IterMethodCheck07:
    def __iter__(self):
        return HasNextButNoIter() # Noncompliant

class HasNextAndIter:
    def __iter__(self): # Noncompliant
        ...

    def __next__(self):
        ...

class IterMethodCheck08:
    def __iter__(self):
        return HasNextAndIter() # Compliant

class IterMethodCheck09:
    def __iter__(self):
        if unknown():
            x = True
        else:
            x = iter(())

        return x # Compliant, potential FN

class IterMethodCheck10:
    def __iter__(self):
        if unknown():
            x = True
        else:
            x = 42

        return x # Noncompliant

class IterMethodCheck11:
    # Does not return anything
    def __iter__(self): # Noncompliant {{Return an object complying with iterator protocol. Consider explicitly raising a NotImplementedError if this class is not (yet) meant to support this method.}}
#   ^^^^^^^^^^^^^^^^^^^
        pass

class IterMethodCheck12:
    def __iter__(self): # Noncompliant
        1/0 # Users actually sometimes create implementations like this to disable special methods

class IterMethodCheck13:
    def __iter__(self):
        return # Noncompliant {{Return an object complying with iterator protocol.}}
    #   ^^^^^^

class IterMethodCheck14:
    def __iter__(self):
        if unknown():
            return iter(())
        # Compliant: Potential FN if unknown() ever resolves to False

class IterMethodCheck15:
    def __iter__(self):
        x = string_unknown()
        if x == 'Empty':
            x = iter(())
        elif x.startswith("["):
            x = iter(x[1:-1].split())

        return x # Compliant, potential FN if string_unknown() returns a string in an unexpected format

class IterMethodCheck16:
    def __iter__(self):
        yield True # Compliant: Generators have the iterator methods

class IterMethodCheck17:
    def __iter__(self):
        yield True # Compliant
        return False # Return inside generators produce StopIteration exceptions

class IterMethodCheck18:
    def __iter__(self):
        (yield True) # Compliant
        return False

class IterMethodCheck19:
    def __iter__(self):
        if unknown():
            return False # Compliant: This is a generator
        else:
            yield True

class IterMethodCheck20:
    def __iter__(self):
        def sub_function():
            yield True
        # The yield statement is in the sub_function, so __iter__ is not a generator here:
        return False # Noncompliant

class IterMethodCheck21:
    def __iter__(self):
        sub_function = lambda: (yield True)
        return False # Noncompliant

class IterMethodCheck22:
    def __iter__(self):
        def sub_function():
            return 42
        return iter(()) # Compliant

class IterMethodCheck23:
    def __iter__(self):
        def sub_function():
            return iter(())
        return 42 # Noncompliant

class IterMethodCheck24:
    async def __iter__(self): # Noncompliant {{Return an object complying with iterator protocol. The method can not be a coroutine and have the `async` keyword.}}
#   ^^^^^
        return iter(())

class IterMethodCheck25:
    def __iter__(self):
        async def sub_function():
            ...

        return iter(()) # Compliant

class IterMethodCheck26:
    def __iter__(self):
        if False:
            # We can not check for dead code. Also, even if a code block is dead, the user might want to be warned about nonsensical return values in special methods.
            return 42 # Noncompliant
        else:
            return iter(())

class IterMethodCheck27:
    def __iter__(self):
        if True:
            return # Noncompliant
        else:
            return iter(())

class RaisesException01:
    def __iter__(self): # Compliant
        raise NotImplementedError()

class RaisesException02:
    def __iter__(self): # Compliant
        raise RuntimeError()

class RaisesException03:
    def __iter__(self): # Compliant
        raise NotImplementedError

class RaisesException04:
    def __iter__(self):
        return 42 # Noncompliant
        raise NotImplementedError()

class RaisesException05:
    def __iter__(self):
        if True:
            return 42 # Noncompliant
        else:
            raise NotImplementedError()

class AbstractIter01(ABC):
    @abstractmethod
    def __iter__(self): # Compliant: The @abstractmethod annotation indicates that the method has not been implemented on purpose
        pass

class AbstractIter02(ABC):
    @some_decorator
    @abc.abstractmethod
    def __iter__(self): # Compliant
        pass

class AbstractIter03(ABC):
    @abstractmethod
    def __iter__(self):
        # Unlike Java abstract methods, python abstract methods can provide an implementation which we should still check
        return "Hello World" # Noncompliant

class AbstractIter04(ABC):
    @abstractmethod
    def __iter__(self): # Compliant
        raise NotImplementedError()

class AbstractIter05(ABC):
    @abstractmethod
    def __iter__(self):
        if True:
            return 42 # Noncompliant
        raise NotImplementedError()

class AbstractIter06(ABC):
    @abstractmethod()
    def __iter__(self): # Compliant
        pass

class AbstractIter07(ABC):
    @unknown_decorator_symbol
    @abstractmethod
    def __iter__(self): # Compliant
        pass

class ProtocolClass01(Protocol):
    def __iter__(self): # Compliant
        ...

class ProtocolClass02(typing.Protocol):
    def __iter__(self): # Compliant
        ...

class ProtocolClass03(Protocol):
    def __iter__(self):
        return 42 # Noncompliant

class ZopeInterfaceClass01(Interface):
    def __iter__(self): # Compliant
        ...

class ZopeInterfaceClass02(zope.interface.Interface):
    def __iter__(self): # Compliant
        ...


def __iter__():
    return True # Compliant: This function is not part of a class definition

class IteratorMissingNext01:
    def __iter__(self):
        return self # Noncompliant

    def next(self):
        ...

class IteratorMissingNext02:
    def __iter__(this, self): # FN
        return self

    def next(self):
        ...

class IteratorMissingNext03:
    def __iter__(this):
        return self # Compliant

    def next(self):
        ...

class IteratorMissingNext04:
    self: Any

    def __iter__(this):
        return self # Compliant

    def next(self):
        ...

class IteratorMissingNext05:
    def __iter__(): # FN
        return self

    def next(self):
        ...

def f():
    self = unknown()
    class IteratorMissingNext06:
        def __iter__(): # FN
            return self

        def next(self):
            ...

class IteratorMissingNext07:
    def __iter__(self):
        return self # FN: Cant resolve class symbol because of reassignment below

    def next(self):
        ...

IteratorMissingNext07 = 42


class IteratorMissingNext07:
    def __iter__(this):
        return this # FN

    def next(self):
        ...

class IteratorWithNext01:
    def __iter__(self):
        return self # Compliant

    def __next__(self):
        ...

class IteratorWithNext02:
    def __iter__(self):
        return self, 42 # Noncompliant

    def __next__(self):
        ...

