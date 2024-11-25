# Noncompliant@+2 {{Remove the unused function parameter "a".}}
'Issues are not raised when the variable is mentioned in a comment related to the function'
def f1(a):
#      ^
    print("foo")


def g1(a):
    print(a)


def f2(a):
    return locals()


def g2(a):  # Noncompliant
    b = 1
    c = 2
    compute(b)

class MyInterface:

    def write_alignment(self, a):
        """This method should be replaced by any derived class to do something
        useful.
        """
        raise NotImplementedError("This object should be subclassed")

class Parent:
    def do_something(self, a, b): # Noncompliant
        #                  ^
        return compute(b)

    def do_something_else(self, a, b):
        return compute(a + b)

    def using_child_method(self):
        return self.method_defined_in_child_class_only()

class Child(Parent):
    def do_something_else(self, a, b):
        return compute(a)

    # SONARPY-2327 `method_defined_in_child_class_only` is considered a member of Parent class, thus S1172 is not raised
    def method_defined_in_child_class_only(self, a):
        return compute()

    def method_defined_in_child_class_only_and_not_used(self, a): # Noncompliant
        #                                                     ^
        return compute()

class AnotherChild(UnknownParent):
    def _private_method(self, a, b):  # OK
        return compute(b)

    def is_overriding_a_parent_method(self, a, b):
        return compute(b)

class ClassWithoutArgs:
    def do_something(self, a, b):  # Noncompliant
        return compute(b)

class ClassWithoutArgs2():
    def do_something(self, a, b):  # Noncompliant
        return compute(b)


@some_decorator
def decorated_function(x, y):
    print("foo")

class MyClass:
    @classmethod
    def foo(cls):
        ...
    def __exit__(self, exc_type, exc_val, exc_tb):
        return exc_val

    def __other(self, x): # Noncompliant
        return 42

def empty_method(x):
    ...

def mapper(x, y):
    return x + 1

for i in map(mapper, [1, 2], [3, 4]):
    print(i)

def some_fun(x, y): # Noncompliant
    return x + 1

some_fun(1, 1)

def ambiguous_f(x): # Noncompliant
    print("foo")

def ambiguous_f(x):
    return x + 1


def not_implemented(r1, r2, r3): # OK
    return NotImplemented

def returning_none(r1): # OK
    return

# coverage

def reassigned_f(x):
    print("hello")

global reassigned_f

def aliased_f(x): # FN
    print("hello")

g = aliased_f
g(42)


class MyFoo:
    def meth(self, x): # Noncompliant
        print("foo")
    def bar(self):
        self.meth(42)


import zope.interface

class IFoo(zope.interface.Interface):
    def bar(q, r=None):
        """bar foo bar"""


def test_using_fixture(my_fixture):
    assert do_something() == expected()


def lambda_handler(_event, _context):
    print("foo")


def no_issue_aws_lambda_parameters(event, context):  # OK, may be required in AWS Lambda context
    print("foo")


class MyClass:
    def param_referenced_in_docstring_no_issue(unused_param):  # OK
       '''
       Overrides may use unused_param to do something
       '''
       print("hello")

    def param_referenced_in_comment_no_issue(unused_param_2):
        # Overrides may use unused_param_2 to do something
        print("hello")

    def param_referenced_in_comment_no_issue_2(unused_param_2):
        # Overrides may use 'unused_param_2' to do something
        print("hello")

    def param_referenced_in_comment_no_issue_3(unused_param_2):
        # Overrides may use "unused_param_2" to do something
        print("hello")

    def param_accessed_through_pandas_no_issue(sample_df, area_of_interest): # OK
        sample_df.query('area == @area_of_interest').population
        print("hello")

import abc
class SomeAbstractClass1(abc.ABC):
    def execute_test(self, name):
        print("Test")

class AnImplementationOfAnAbstractClass(SomeAbstractClass1):
    def execute_test_nc(self, name): # Noncompliant
        print("Test")

 class SomeClassWithMetaclass1(metaclass=abc.ABCMeta):
    def execute_suite(self, name):
        print("Suite")

import abc as xyz
class SomeClassWithMetaclass2(metaclass=xyz.ABCMeta):
    def execute_suite(self, name):
        print("Suite")
class SomeAbstractClass2(xyz.ABC):
    def execute_test(self, name):
        print("Test")

from abc import ABC as XYZ
class SomeAbstractClass3(XYZ):
    def execute_test(self, name):
        print("Test")

class FakeMetaclass(type):
    ...

class ClassWithFakeMetaClass(metaclass=FakeMetaclass):
    def execute(self, name):
        print("Execute")


from typing import Callable
class LocalClassWithAnnotatedMember:
  my_member: Callable[[str, int],str]

class LocalClassChild(LocalClassWithAnnotatedMember):
  def my_member(self, param, other_param): # OK, respecting contract defined in parent
    print("Execute")
