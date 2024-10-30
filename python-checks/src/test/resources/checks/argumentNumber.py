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
    def meth(p1, p2): pass
    class A:
      def __new__(cls, a, b):
          cls.__new__(A, 1, 2)
      def meth(self, p1, p2):
        meth(1, 2) # OK
        meth(1, 2, 3) # Noncompliant
      @classmethod
      def class_meth(cls, p1, p2): pass
      @staticmethod
      def static_meth(p1, p2): pass
      def foo(p1): pass
      foo(42)
      @classmethod
      def bar_class_method(cls):
        cls.bar_instance_method(cls)
      def bar_instance_method(self): pass

    A.class_meth(42) # FN {{'class_meth' expects 2 positional arguments, but 1 was provided}}

    A.class_meth(1, 2) # OK
    A.static_meth(42) # FN {{'static_meth' expects 2 positional arguments, but 1 was provided}}

    A.static_meth(1, 2) # OK
    a = A()
    a.meth(42, 43)
    a.meth(42) # Noncompliant {{Add 1 missing arguments; 'meth' expects 2 positional arguments.}}
    a.meth(42, 43, 44) # Noncompliant {{Remove 1 unexpected arguments; 'meth' expects 2 positional arguments.}}

    A.foo() # FN
    A.foo(42)

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
    myList.sort(lambda x: x)

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


class BuiltinFunctionWithEmptyParameterName:
    def __init__(self, name, value):
        setattr(self, name, value)  # OK


class MyTZInfo(datetime.tzinfo):
    def tzname(self):  # FN
        pass


def logging_api():
    logging.basicConfig(format="42", force=True)  # OK

def foo(day, tz):
    b = datetime.date.fromordinal(day).replace(tzinfo=tz) # FN SONARPY-1472
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
