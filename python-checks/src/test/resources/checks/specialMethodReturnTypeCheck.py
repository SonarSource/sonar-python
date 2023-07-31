from external import unknown
from abc import ABC, abstractmethod
import abc
from some_module import some_decorator

def string_unknown() -> str:
    ...

def bool_unknown() -> bool:
    ...

class BoolMethodCheck01:
    def __bool__(self):
        return "Hello" # Noncompliant {{Return a value of type `bool` here.}}
        #      ^^^^^^^

class BoolMethodCheck02:
    def __bool__(self):
        return True # Compliant

class BoolMethodCheck03:
    def __bool__(self):
        x = "Hello"
        return x # Noncompliant

class BoolMethodCheck04:
    def __init__(self, condition):
        self.condition = condition

    def __bool__(self):
        return self.condition # Compliant but potential for FNs: Technically, it is possible that this may not be a boolean but there is no explicit indication for it. Marking this may lead to FPs

class BoolMethodCheck05:
    def __init__(self, condition):
        self.condition = condition

    def __bool__(self):
        return self.condition # FN (secondary location): See guaranteed invocation below

b = BoolMethodCheck05("Hello")
bool(b) # FN

class BoolMethodCheck06:
    def __bool__(self):
        if unknown():
            x = "Hello"
        else:
            x = True

        return x # Compliant, potential FN

class BoolMethodCheck07:
    def __bool__(self):
        if unknown():
            x = "Hello"
        else:
            x = 42

        return x # Noncompliant

class BoolMethodCheck08:
    # Does not return anything
    def __bool__(self): # Noncompliant {{Return a value of type `bool` in this method. Consider explicitly raising a TypeError if this class is not meant to support this method.}}
#   ^^^^^^^^^^^^^^^^^^^
        pass

class BoolMethodCheck09:
    def __bool__(self): # Noncompliant
        1/0 # Users actually sometimes create implementations like this to disable special methods

class BoolMethodCheck10:
    def __bool__(self):
        return # Noncompliant {{Return a value of type `bool` in this method.}}
    #   ^^^^^^

class BoolMethodCheck11:
    # Does not return anything
    def __bool__(self):
        if unknown():
            return True
        # Compliant: Potential FN if unknown() ever resolves to False

class BoolMethodCheck12:
    def __bool__(self):
        x = string_unknown()
        if x == 'True':
            x = True
        elif x == 'False':
            x = False

        return x # Compliant, potential FN

class BoolMethodCheck13:
    def __bool__(self):
        yield True # Noncompliant {{Return a value of type `bool` in this method. The method can not be a generator and contain `yield` expressions.}}
#       ^^^^^

class BoolMethodCheck14:
    def __bool__(self):
        yield True # Noncompliant
        return False

class BoolMethodCheck15:
    def __bool__(self):
        (yield True) # Noncompliant
    #    ^^^^^
        return False

class BoolMethodCheck16:
    def __bool__(self):
        if unknown():
            return False
        else:
            yield True # Noncompliant

class BoolMethodCheck17:
    def __bool__(self):
        def sub_function():
            yield True # Compliant
        return False

class BoolMethodCheck18:
    def __bool__(self):
        sub_function = lambda: (yield True) # Compliant
        return False

class BoolMethodCheck19:
    def __bool__(self):
        def sub_function():
            return 42 # Compliant
        return False

class BoolMethodCheck20:
    async def __bool__(self): # Noncompliant {{Return a value of type `bool` in this method. The method can not be a coroutine and have the `async` keyword.}}
#   ^^^^^
        return False

class BoolMethodCheck21:
    def __bool__(self):
        async def sub_function(): # Compliant
            ...

        return False

class BoolMethodCheck22:
    def __bool__(self):
        if False:
            # We can not check for dead code. Also, even if a code block is dead the user might want to be warned about nonsensical return values in special methods.
            return "Hello" # Noncompliant
        else:
            return False

class BoolMethodCheck23:
    def __bool__(self):
        if True:
            return # Noncompliant
        else:
            return False

class IndexMethodCheck01:
    def __index__(self):
        return "Hello" # Noncompliant

class IndexMethodCheck02:
    def __index__(self):
        return 42 # Compliant

class ReprMethodCheck01:
    def __repr__(self):
        return 42 # Noncompliant

class ReprMethodCheck02:
    def __repr__(self):
        return "Hello" # Compliant

class StrMethodCheck01:
    def __str__(self):
        return 42 # Noncompliant

class StrMethodCheck02:
    def __str__(self):
        return "Hello" # Compliant

class text(str):
  pass
class StrMethodCheck03:
    def __str__(self):
        return text("Hello") # Compliant: Although the python documentation specifies that the value "must be a string object", no type error is thrown if type subclasses are used

class StrMethodCheck04:
    def __str__(self):
        return b'Hello' # FN: The type analysis assigns the Any type to bytes literals

class StrMethodCheck05:
    def __str__(self):
        # This encode call will return a bytes object, not a string
        return 'Hello'.encode('utf-8') # Noncompliant

class BytesMethodCheck01:
    def __bytes__(self):
        return 42 # Noncompliant

class BytesMethodCheck02:
    def __bytes__(self):
            return b'A' # Compliant

class BytesMethodCheck03:
    def __bytes__(self):
            return bytes(10) # Compliant

class HashMethodCheck01:
    def __hash__(self):
        return "Hello" # Noncompliant

class HashMethodCheck02:
    def __hash__(self):
        return 42 # Compliant

class HashMethodCheck03:
    def __hash__(self):
        return 42.1 # Noncompliant

class FormatMethodCheck01:
    def __format__(self, format_spec):
        return 42 # Noncompliant

class FormatMethodCheck02:
    def __format__(self, format_spec):
        return "Hello" # Compliant


class GetNewArgs01:
    def __getnewargs__(self):
        return ("Hello World", 42) # Compliant

class GetNewArgs02:
    def __getnewargs__(self):
        return "Hello World", 42 # Compliant

class GetNewArgs03:
    def __getnewargs__(self):
        return 42 # Noncompliant {{Return a value of type `tuple` here.}}
        #      ^^

class GetNewArgs04:
    def __getnewargs__(self):
        if True:
            return "Hello World", 42 # Compliant
        else:
            return 42 # Noncompliant

class GetNewArgsEx01:
    def __getnewargs_ex__(self):
        return ("Hello World", 42), {} # Compliant

class GetNewArgsEx02:
    def __getnewargs_ex__(self):
        return (("Hello World", 42), {}) # Compliant

class GetNewArgsEx03:
    def __getnewargs_ex__(self):
        return (42, {}) # Noncompliant {{Return a value of type `tuple[tuple, dict]` here.}}
        #       ^^^^^^

class GetNewArgsEx04:
    def __getnewargs_ex__(self):
        return ("Hello World", 42), 10 # Noncompliant {{Return a value of type `tuple[tuple, dict]` here.}}
        #      ^^^^^^^^^^^^^^^^^^^^^^^

class GetNewArgsEx05:
    def __getnewargs_ex__(self):
        return (10, ) # Noncompliant {{Return a value of type `tuple[tuple, dict]` here. A tuple of two elements was expected but found tuple with 1 element(s).}}

class GetNewArgsEx06:
    def __getnewargs_ex__(self):
        return ("Hello World", 42), 1337, {} # Noncompliant {{Return a value of type `tuple[tuple, dict]` here. A tuple of two elements was expected but found tuple with 3 element(s).}}
        #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class GetNewArgsEx07:
    def __getnewargs_ex__(self):
        return 1337, ("Hello World", 42), {} # Noncompliant

class GetNewArgsEx08:
    def __getnewargs_ex__(self):
        return (("Hello World", 42), 1337, {}) # Noncompliant
        #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class GetNewArgsEx09:
    def __getnewargs_ex__(self):
        x = (("Hello World", 42), 1337, {})
        return x # FN

class GetNewArgsEx10:
    def __getnewargs_ex__(self):
        if unknown():
            x = {}
        else:
            x = 42

        x = (("Hello World", 42), x)
        return x # Compliant, but potential FN if unknown() evaluates to False
class RaisesException01:
    def __hash__(self): # Compliant
        raise TypeError("unhashable type: RaisesException01")

class RaisesException02:
    def __hash__(self): # Compliant
        raise NotImplementedError()

class RaisesException03:
    def __hash__(self): # Compliant
        raise NotImplementedError

class RaisesException04:
    def __hash__(self):
        return "Hello World" # Noncompliant
        raise NotImplementedError()

class RaisesException05:
    def __hash__(self):
        if True:
            return "Hello World" # Noncompliant
        else:
            raise NotImplementedError()

class RaisesException06:
    def __hash__(self):
        if True:
            return 42
        else:
            raise NotImplementedError() # Compliant

class AbstractSpecialMethod01(ABC):
    @abstractmethod
    def __bool__(self): # Compliant: The @abstractmethod annotation indicates that the method has not been implemented on purpose
        pass

    @some_decorator
    @abc.abstractmethod
    def __index__(self): # Compliant
        pass

    @abstractmethod
    def __hash__(self):
        # Unlike Java abstract methods, python abstract methods can provide an implementation which we should still check
        return "Hello World" # Noncompliant

    @abstractmethod
    def __str__(self): # Compliant
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        if True:
            return 42 # Noncompliant
        raise NotImplementedError()

    @abstractmethod()
    def __bytes__(self): # Compliant
        pass

    @unknown_decorator_symbol
    @abstractmethod
    def __format__(self, format_spec): # Compliant
        pass

def __bool__():
    return 42 # Compliant: This function is not part of a class definition
