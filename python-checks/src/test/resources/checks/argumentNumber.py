import zope
from argumentNumberImported import fn
from contextvars import copy_context
import datetime
import logging

fn(1, 2) # OK, no project level information

def functions():

    def foo(p1,p2): pass

    foo(42) # Noncompliant {{Add 1 missing arguments; 'foo' expects 2 positional arguments.}}
#   ^^^
    foo(p1=42) # Noncompliant {{Add 1 missing arguments; 'foo' expects 2 positional arguments.}}

    foo(1,2,3) # Noncompliant {{Remove 1 unexpected arguments; 'foo' expects 2 positional arguments.}}
#   ^^^
    foo(1, x = 42) # Noncompliant {{Add 1 missing arguments; 'foo' expects 2 positional arguments.}} {{Remove this unexpected named argument 'x'.}}
    foo(1, 2, x = 42) # Noncompliant {{Remove this unexpected named argument 'x'.}}
#             ^^^^^^

    foo(1, 2) # OK
    foo(1, p2 = 2) # OK
    args = [1,2]
    foo(*args) # OK


    def foo_with_default_value(p1, p2 = 42): pass

    foo_with_default_value() # Noncompliant {{Add 1 missing arguments; 'foo_with_default_value' expects at least 1 positional arguments.}}
#   ^^^^^^^^^^^^^^^^^^^^^^
    foo_with_default_value(42) # OK
    foo_with_default_value(1,2,3) # Noncompliant {{Remove 1 unexpected arguments; 'foo_with_default_value' expects at most 2 positional arguments.}}

    def default_values(p1, p2 = 42, p3 = 43): pass
    default_values(1)
    default_values(1, 2)
    default_values(1, 2, 3)

    if True:
        def foo_with_multiple_binding(p1, p2): pass
    else:
        def foo_with_multiple_binding(p1): pass

    foo_with_multiple_binding(1) # OK

    def foo_with_keyword_only(p1, *, p2): pass

    foo_with_keyword_only() # Noncompliant {{Add 1 missing arguments; 'foo_with_keyword_only' expects 1 positional arguments.}} {{Add the missing keyword arguments: 'p2'}}
#   ^^^^^^^^^^^^^^^^^^^^^
    foo_with_keyword_only(42) # Noncompliant {{Add the missing keyword arguments: 'p2'}}
#   ^^^^^^^^^^^^^^^^^^^^^
    foo_with_keyword_only(1, 2, p2 = 3) # Noncompliant {{Remove 1 unexpected arguments; 'foo_with_keyword_only' expects 1 positional arguments.}}

    def foo_with_variadics(**kwargs): pass

    foo_with_variadics(1, 2, 3) # OK

    def foo_with_optional_keyword_only(p1, *, p2 = 42): pass
    foo_with_optional_keyword_only() # Noncompliant {{Add 1 missing arguments; 'foo_with_optional_keyword_only' expects 1 positional arguments.}}
    foo_with_optional_keyword_only(42) # OK
    foo_with_optional_keyword_only(42, 43) # Noncompliant {{Remove 1 unexpected arguments; 'foo_with_optional_keyword_only' expects 1 positional arguments.}}
    foo_with_optional_keyword_only(42, p2 = 43)

    from mod import dec
    @dec
    def foo_with_decorator(): pass

    foo_with_decorator(1, 2, 3) # OK

    def python2_tuple_params(p1, (p2, p3)): pass
    python2_tuple_params(1, something)
    python2_tuple_params(1, (2, 3))
    python2_tuple_params(1, (2, 3), 4) # Noncompliant
    def python2_tuple_params2((p1, p2), (p3, p4)): pass
    python2_tuple_params2(x) # Noncompliant
    python2_tuple_params2(x, y)
    python2_tuple_params2(x, y, z) # Noncompliant

def methods():
    def meth(p1, p2, p3, p4): pass
    class A:
        def __new__(cls, a, b):
            cls.__new__(A, 1, 2)
        def meth(self, p1, p2):
            meth(1, 2) # Noncompliant 
            meth(1, 2, 3, 5) # oK

        def bar(self, p1, p2): pass

        def other_method(self):
            self.bar(1, 2) 
            self.other_method()

        @classmethod
        def class_meth(cls, p1, p2): pass
        @staticmethod
        def static_meth(p1, p2): pass
        def foo(): pass

        @classmethod
        def bar_class_method(cls):
            cls.bar_instance_method(cls)
            cls.bar_instance_method(cls, 1) # Noncompliant
        def bar_instance_method(self): pass

    A.class_meth(42) # FN {{'class_meth' expects 2 positional arguments, but 1 was provided}}

    A.class_meth(1, 2) # OK
    A.static_meth(42) # FN {{'static_meth' expects 2 positional arguments, but 1 was provided}}

    A.static_meth(1, 2) # OK

    a = A()
    a.meth(42, 43)
    a.meth(42) # Noncompliant {{Add 1 missing arguments; 'meth' expects 2 positional arguments.}}
    a.meth(42, 43, 44) # Noncompliant {{Remove 1 unexpected arguments; 'meth' expects 2 positional arguments.}}

    a.other_method() # OK

    A.bar(42, 43) # Noncompliant {{Add 1 missing arguments; 'bar' expects 3 positional arguments.}}
    A.bar(a, 42, 53)


    A.foo() 
    A.foo(42) # Noncompliant {{Remove 1 unexpected arguments; 'foo' expects 0 positional arguments.}}

    a.foo()
    a.foo(42) # Noncompliant {{Remove 1 unexpected arguments; 'foo' expects 0 positional arguments.}}

    m = a.meth
    m(42, 43) # OK
    m(42) # FN

    class MyInterface(zope.interface.Interface):
        def foo(): pass
    x = MyInterface()
    x.foo()


    # Coverage: loop in inheritance
    class A1(A2):
      def fn(self): pass

    class A2(A3):
      pass

    class A3(A1):
      pass

    a1 = A1()
    a1.fn(42) # Noncompliant


    class B1:
      def __reduce__(self, p1, p2):
        pass

    class B2(B1):
      def foo(self):
        super().__reduce__(1, 2) # OK, __reduce__ is not 'object.__reduce__' but B1.__reduce__
        super().__reduce__(1, 2, 3) # Noncompliant

    class MethodsWithKeywordArguments:
        @classmethod
        def from_dict(cls, **kwargs):
            kwargs.items() 
            kwargs.items(1) # Noncompliant


     
    class MethodsWithTryExcept(object):
        def get_all_config(self):
            pass

        def replace(self, config):
            try:
                pass
            except Exception as e:
                pass
            self.get_all_config() 
            self.get_all_config(3) # Noncompliant

            self.replace(3) # OK
            self.replace(3, 4) # Noncompliant
            MethodsWithTryExcept.replace(self, 4) # OK
            MethodsWithTryExcept.replace(4) # Noncompliant

    class MethodsWithControlFlow:
        def something(self, importing=None):
            if importing is None:
                importing = set()
            importing.discard(1)
            importing.discard(1, 2) # FN

def methods_on_types():
    class Test: pass

    sys_type = type(Test)
    sys_type.__setattr__(sys, key,  "42") # Ok
    sys_type.__setattr__(key, "42") # FN see SONARPY-3591

    type.mro() # Noncompliant
    type.mro(type) # Ok

    sys_type.mro() # FN see SONARPY-3591
    sys_type.mro(Test) # Ok

    test_instance = Test()
    test_instance_type = type(test_instance)
    test_instance_type.mro() # Ok
    test_instance_type.mro(test_instance) #  see SONARPY-3591


    import typing
    _T = typing.TypeVar('_T')

    class GenericTest(typing.Generic[_T]):
        cls: _T
        def __new__(cls): 
            pass

        def test(self):
            GenericTest.cls.__new__(self.cls) # Ok
            GenericTest.cls.__new__(self.cls, 1) # FN
   

def builtin_method():
    myList = list(42, 43)
    myList.append(44)
    myList.append(1, 2) # Noncompliant
    l = list[int](42, 43)
    l.append(1, 2) # Noncompliant
    import math
    math.acos(0, 1) # Noncompliant

def builtin_method_different_for_python_2_and_3():
    myList = list(42, 43)
    myList.sort()
    myList.sort(lambda x: x)  # Noncompliant

def typeshed_third_party_methods():
  copy_context(42) # Noncompliant


def no_overlap_with_S5549():
  def keyword_only(a, *, b): ...
  def positional_only(a, /, b): ...

  keyword_only(1, b=2, a=2)  # S5549 scope
  positional_only(1, 2, b=2)  # S5549 scope
  positional_only(1, 2, a=2)  # Noncompliant
  positional_only(1, a=2)  # Noncompliant 2

  class MyClass:
    def method1(self, a): ...
    def method2(self, a):
      self.method1(self, a)  # Noncompliant
      self.method1(a, self=self) # S5549 scope
      self.method1(self, a=a) # S5549 scope
      "{self}".format(self=self)  # Ok



def flask_send_file():
    # make sure no FPs are raised on flask.send_file
    from flask import send_file
    return send_file(
        status.message,
        mimetype=APPLICATION_MIME_TYPE,
        as_attachment=True,
        download_name=f"{analytics.filename}.zip",
    )


# Fixing FPs reported in SONARPY-872
def jinja_apis():
    from jinja2.filters import do_indent, do_wordwrap
    do_wordwrap(environment, s, break_on_hyphens=False)
    do_indent(s, first=first, blank=blank)


def metaclass_new():
    class ModelBase(type):
        def __new__(cls, name, bases, attrs, **kwargs):
            super_new = super().__new__
            return super_new(cls, name, bases, attrs)

    class ExplicitModelBase(type):
        def __new__(cls, name, bases, attrs, **kwargs):
            super_new = super(ExplicitModelBase, cls).__new__
            return super_new(cls, name, bases, attrs)

    class ParentModelBase(type):
        def __new__(cls, name, bases, attrs):
            return super().__new__(cls, name, bases, attrs)

    class ChildModelBase(ParentModelBase):
        def __new__(cls, name, bases, attrs, **kwargs):
            super_new = super().__new__
            return super_new(cls, name, bases, attrs)


class BuiltinFunctionWithEmptyParameterName:
    def __init__(self, name, value):
        setattr(self, name, value)  # OK


class MyTZInfo(datetime.tzinfo):
    def tzname(self):  # FN
        pass


def logging_api():
    logging.basicConfig(format="42", force=True)  # OK

def foo(day, tz):
    b = datetime.date.fromordinal(day).replace(tzinfo=tz) # Noncompliant
    a = datetime.datetime.fromordinal(day).replace(tzinfo=tz) # OK


def bound_and_unbound_methods():
    class ClassWithMethod:
        def some_method(self, param): ...

    unbound_method = ClassWithMethod.some_method
    bound_method = ClassWithMethod().some_method

    unbound_method(1, 2) # OK
    bound_method(1) # OK

    unbound_method() # FN SONARPY-2285
    unbound_method(1) # FN SONARPY-2285
    unbound_method(1, 2, 3) # FN SONARPY-2285
    bound_method() # FN SONARPY-2285
    bound_method(1, 2) # FN SONARPY-2285

def torch():
    import torch
    torch.cat(torch.Tensor(), axis=-1)

def pandas():
    import pandas as pd
    pd.testing.assert_index_equal(left_index, right_index, check_names=False)
    df = pd.DataFrame()
    df.stack("MASTER_FND_CODE", future_stack=True)


def aliased_super_reassignment():
    class Base(type):
        def __new__(cls, name, bases, attrs):
            factory = None
            factory = super().__new__
            return factory(cls, name, bases, attrs)


def aliased_new_without_reaching_definitions():
    class Meta(type):
        def __new__(cls, name, bases, attrs):
            factory = super().__new__
            try:
                attrs = dict(attrs)
            except ValueError:
                pass
            return factory(cls, name, bases, attrs)

    class Ordinary:
        def __new__(cls):
            return object.__new__(cls)
    factory = Ordinary.__new__
    try:
        factory(Ordinary, 42)  # Noncompliant
    except ValueError:
        pass


def callable_alternatives(condition, unknown):
    def left(value, *, left=False):
#       ^^^^> {{Function definition.}}
        pass
    def right(value, *, right=False):
#       ^^^^^> {{Function definition.}}
        pass
    either = left if condition else right
    either(1, left=True)
    either(1, right=True)
    either(1, nonexistent=True)  # Noncompliant {{Remove this unexpected named argument 'nonexistent'.}}
    #         ^^^^^^^^^^^^^^^^
    either()  # Noncompliant {{Add 1 missing arguments; this callable expects 1 positional arguments.}}
    either(1, 2)  # Noncompliant {{Remove 1 unexpected arguments; this callable expects 1 positional arguments.}}

    def optional_left(value, optional=0):
        pass
    def optional_right(value, optional=0, extra=0):
        pass
    same = optional_left if condition else optional_right
    same()  # Noncompliant {{Add 1 missing arguments; this callable expects at least 1 positional arguments.}}
   #^^^^
    same(1, 2, 3)  # Valid for one alternative.
    same(1, nonexistent=True)  # Noncompliant
    same(*unknown)
    same(**unknown)

    def exact(value):
        pass
    shared_minimum = exact if condition else optional_left
    shared_minimum()  # Noncompliant {{Add 1 missing arguments; this callable expects at least 1 positional arguments.}}
   #^^^^^^^^^^^^^^
    shared_minimum(1)

    def required(value, optional):
        pass
    shared_maximum = required if condition else optional_left
    shared_maximum(1, 2, 3)  # Noncompliant {{Remove 1 unexpected arguments; this callable expects at most 2 positional arguments.}}
   #^^^^^^^^^^^^^^
    shared_maximum(1, 2)

    def variadic(*args, **kwargs):
        pass
    flexible = left if condition else variadic
    flexible(1, nonexistent=True)
    uncertain = left if condition else unknown
    uncertain(1, nonexistent=True)

    class Callable:
        def method(self, value):
            pass
    maybe_class = left if condition else Callable
    maybe_class(1, nonexistent=True)
    maybe_bound = left if condition else Callable().method
    maybe_bound(1, nonexistent=True)

    @unknown
    def decorated(value):
        pass
    maybe_decorated = left if condition else decorated
    maybe_decorated(1, nonexistent=True)

    def required_left(*, shared, left):
        pass
    def required_right(*, shared, right):
        pass
    keywords = required_left if condition else required_right
    keywords()  # Noncompliant {{Add the missing keyword arguments: 'shared'}}
    keywords(left=1, right=1)  # Noncompliant {{Add the missing keyword arguments: 'shared'}}

    def reordered_left(*, second, first):
        pass
    def reordered_right(*, first, second):
        pass
    reordered = reordered_left if condition else reordered_right
    reordered()  # Noncompliant {{Add the missing keyword arguments: 'first' 'second'}}
    reordered(first=1, second=2)

    def no_arguments():
        pass
    def two_arguments(first, second):
        pass
    conflicting = no_arguments if condition else two_arguments
    conflicting(1)  # One alternative needs an argument added, the other needs it removed.
    conflicting(1, 2, 3)  # Different numbers of excess arguments.
    conflicting()  # Valid for one alternative.


def library_signatures(condition):
    import subprocess
    # Old descriptors omit these forwarded keywords. Library overloads remain outside this change.
    subprocess.check_call(["echo", "ok"], encoding="utf-8", errors="replace")
    subprocess.call(["echo", "ok"], encoding="utf-8", errors="replace")

    def local(args):
        pass
    mixed = local if condition else subprocess.check_call
    mixed([], encoding="utf-8")

    import math
    # Single library signatures remain checked.
    math.sqrt()  # Noncompliant


# super() with incomplete ancestry: resolve local members of the sole direct parent.
from unknown import UnknownBase, unknown_decorator


class Parent(UnknownBase):
    def __init__(self, mod, symmetric=True):
        pass

    def method(self, value):
        pass

    def __new__(cls, value):
        return object.__new__(cls)


class Child(Parent):
    def __init__(self, mod, symmetric=True):
        super().__init__(mod, object(), symmetric)  # Noncompliant {{Remove 1 unexpected arguments; '__init__' expects at most 2 positional arguments.}}
        super().__init__(mod, symmetric)
        super().__init__(mod, symmetric=symmetric)
        super().__init__()  # Noncompliant {{Add 1 missing arguments; '__init__' expects at least 1 positional arguments.}}
        super().method()  # Noncompliant
        super().method(mod)
        super().method(mod, unknown=True)  # Noncompliant
        super().missing(mod, object(), symmetric)  # Unknown member stays unknown.
        super(Child, self).__init__(mod, object(), symmetric)  # Explicit super remains conservative.

    def __new__(cls, value):
        return super().__new__(cls, value)  # __new__ is implicitly static; cls is explicit.


class Intermediate(Parent):
    pass


class Grandchild(Intermediate):
    def __init__(self):
        super().__init__(1, 2, 3)  # Do not search past a direct base without a local member.


class Other:
    def __init__(self, first, second, third):
        pass


class MultipleBases(Parent, Other):
    def __init__(self):
        super().__init__(1, 2, 3)  # Incomplete multiple-inheritance MRO remains unknown.


class UnknownFirstBase(UnknownBase, Other):
    def __init__(self):
        super().__init__(1, 2, 3, 4)  # Do not skip an unknown leading base.


class DecoratedMethod(UnknownBase):
    @unknown_decorator
    def method(self, value):
        pass


class DecoratedChild(DecoratedMethod):
    def run(self):
        super().method()  # Arbitrary decorators may change the signature.


class VariadicParent(UnknownBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Cooperative forwarding stays conservative.


class VariadicChild(VariadicParent):
    def __init__(self):
        super().__init__(1, 2, 3, extra=4)


class Root:
    def method(self, value):
        pass


class Left(Root):
    pass


class Right(Root):
    def method(self, first, second):
        pass


class CompleteDiamond(Left, Right):
    def run(self):
        super().method(1, 2)  # Preserve C3 lookup of the sibling override.
        super().method(1)  # Noncompliant


class InvalidMro(Root, Left):
    def run(self):
        super().method(1, 2, 3)  # Fully resolved but inconsistent MRO remains unknown.



def nominal_super_calls_with_incomplete_ancestry():
    from unknown import UnknownBase

    class Parent(UnknownBase):
        def __init__(self, mod, symmetric=True):
            pass

        def method(self, value):
            pass

    class NonFinalConstructor(Parent):
        def __init__(self, mod, symmetric=True):
            super().__init__(mod, object(), symmetric)  # Noncompliant

    class Child(Parent):
        def run(self):
            # FP: inference uses Child's MRO, although Sibling precedes Parent on Concrete.
            super().method(1, 2)  # Noncompliant

    class Sibling(Parent):
        def method(self, first, second):
            pass

    class Concrete(Child, Sibling):
        pass

    Concrete(1).run()


def super_with_nested_final_decorator():
    from typing import final
    from unknown import UnknownBase

    class Parent(UnknownBase):
        def method(self, value):
            pass

    @final
    class Child(Parent):
        def run(self):
            super().method(1, 2)  # Noncompliant
