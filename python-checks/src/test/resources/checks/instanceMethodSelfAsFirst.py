class Foo:
    def bar(some_argument): # Noncompliant {{Rename "some_argument" to "self" or add the missing "self" parameter.}}
           #^^^^^^^^^^^^^
        pass

    def baz(self, some_argument):
        pass

    def empty_paramlist():
        pass

    # Known exceptions
    def __init_subclass__(cls, x, y):
        pass

    def __class_getitem__(cls, x, y):
        pass

    def __new__(cls, x, y):
        pass

    def packed(*args, **kwargs): # Do not raise if the first argument is a parameter pack
        pass

    def underline(_, a, b): # Do not raise on "_"
        pass

    @staticmethod
    def staticmethod(a, b, c):
        pass

    @classmethod
    def classmethod(a, b, c):
        pass

    @otherdecorator
    def not_class_or_static(first, b, c): # Noncompliant
                           #^^^^^
        pass

    # Exceptions with "cls" or "mcs"
    @otherdecorator
    def has_decorator_cls(cls, b, c): # OK
        pass

    # Old-style decorators
    def old_style_class_method(cls):
        pass
    old_style_class_method = classmethod(old_style_class_method)

    def old_style_class_method_two(cls, arg):
        return arg
    old_style_class_method_two = classmethod(old_style_class_method_two)

    def old_style_static_method(arg):
        return arg
    old_style_static_method = staticmethod(old_style_static_method)

    # Do not raise on nested classes
    class NestedInClass:
        def nested_method(nested_self):
            pass

    def _called_in_cls_body(x): # OK, used in a call in the class body
        return 1

    some_prop = _called_in_cls_body(5)

    def referenced_in_cls_body(x): # OK
        return 1

    options = [referenced_in_cls_body]

    def used_as_decorator(method):  # OK
        return method

    @used_as_decorator
    def decorated(self): ...

    def _private_method(x): # Noncompliant
        return x

    def calls(self):
        y = _private_method(0)

    def redefined_function(x):
        pass

    redefined_function = 1

def empty_function():
    pass

def free_function(x):
    f = Foo()
    f._private_method(x)
    Foo._private_method(x)

import zope
import zope.interface as zi

class MyInterface(zope.interface.Interface):
    def mymethod(x, y): # Do not raise on zope interfaces
        pass

class MyDerivedInterface(MyInterface):
    def anothermethod(a, b):
        pass

class AnotherInterface(zi.Interface):
    def mymethod(x, y):
        pass

class MetaClass(type):
    def foo(cls):
        pass

import typing.Protocol

class AnotherMetaClass(typing.Protocol):
    def foo(cls):
        pass

@some_decorator
class MightBeMetaclass():
    def foo(mcs):
        pass

import django.utils.decorators.classproperty
import django.utils.decorators as dud

class ClassProperty:
    @classproperty
    def prop1(cls, x):
        pass

    @django.utils.decorators.classproperty
    def prop2(cls, x):
        pass

    @dud.classproperty
    def prop3(cls, x):
        pass

    def no_decorator(cls, x): # Noncompliant
        pass

class EdgeCases:
    def __init__(self, p):
        self.p = p

    def a_method(some_argument): # FN
        pass

    a_method = free_function(a_method)
    tup = (a_method)
    dummy = no_such_func(a_method)
    another = classmethod(*tup)
    dummy2 = classmethod(empty_function())

    def tuple_params((x, y)):
        pass

    def star_token(*, x, y):
        pass

    if p:
        def then_method(self, x): return 1
    else:
        def else_method(arg): # Noncompliant
                       #^^^
            return 2

class AnotherEdgeCase:
    def foo(an_argument): # FN
        pass

AnotherEdgeCase = 1

def used_as_decorator_call(method):  # OK
  return method

@used_as_decorator_call()
def decorated(self): ...
