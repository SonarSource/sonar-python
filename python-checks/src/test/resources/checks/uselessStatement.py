import math
import numpy

class MyClass:
    attr = 42

    def method(self): ...

    @classmethod
    def class_method(cls): ...

    @staticmethod
    def static_method(): ...

    @property
    def my_property(self): ...

class CustomException(TypeError): ...

def a_function(): ...

async def statements_having_effects_or_meant_to_be_ignored(param):
    ...
    pass
    await param
    param()
    x = lambda: param
    (lambda: 2)()
    yield 42
    return 1

def literals():
    True  # Noncompliant {{Remove or refactor this statement; it has no side effects.}}
#   ^^^^
    False  # Noncompliant
    None  # Noncompliant
    1  # Noncompliant
    1.1  # Noncompliant
    [1, 2]  # Noncompliant
    {1, 2}  # Noncompliant
    {1: 1, 2: 2}  # Noncompliant
    "str"  # OK (reportOnStrings false by default)

def unassigned_expressions(param):
    x = 3
    param  # Noncompliant
    x  # Noncompliant
    lambda: 2  # Noncompliant
    func(42); param  # Noncompliant

def conditional_expressions():
    1 if True else 2  # Noncompliant
    1 if True else func()  # OK
    foo() if True else None  # OK
    None if True else foo()  # OK
    if True:
        1  # Noncompliant
    else:
        func()  # Ok

def binary_and_unary_expressions():
  """By default, ignored operators are ">>, <<, |" """
  param < 1  # Noncompliant
  param + param  # Noncompliant
  + param  # Noncompliant
  - param  # Noncompliant
  float16(65472)+float16(32) # Noncompliant
  float16(2**-13)/float16(2) # Noncompliant
  1 / 0 # Noncompliant
  a << b # OK
  a >> b # OK
  a | b  # OK

def expression_list():
  basic.LENGTH, basic.DATA, basic.COMMA, basic.NUMBER # OK (lazy loading)
  something, a_call(), other # Noncompliant 2

def non_called_functions_and_classes():
    round  # Noncompliant
    a_function  # Noncompliant

    MyClass  # Noncompliant
    NotImplemented  # Noncompliant

    # No issue will be raised on Exception classes. This use case is covered by S3984
    BaseException  # Ok
    Exception  # Ok
    ValueError  # Ok
    CustomException  # Ok

def accessing_members(param):
    """To avoid FPs on lazy-loaded attributes, we will only raise on methods which are not annotated as properties"""
    MyClass.attr  # FN
    MyClass.my_property # OK, properties might involve lazy loading
    MyClass.class_method  # Noncompliant
    MyClass.static_method  # Noncompliant
    MyClass.__eq__  # Noncompliant
    instance = MyClass()
    instance.attr # FN
    instance.my_property # OK

def corner_cases():
    try:
        a_function # FN: we avoid raising issues when inside a try/except
    except AttributeError as e:
        pass

    param and param()  # No issue when there are calls within a boolean expression (could be used as an if)
    not (param() or param)
    param and param + 1  # Noncompliant
    not (a and b) # Noncompliant
    call(42) == "hello" # Noncompliant
    [a() for a in param]  #  No issue for comprehensions: might be used as a loop (although it's a code smell)
    a.addCallBack(42).unknown # OK

def invalid_docstring(name):
    class MyClass():
        'Name is %s' % name  # Noncompliant
        ...

def print_statements():
  print "a"
  # In Python2 this is a simple print statement, which is compliant
  # In Python3, this is a binary expression whose lhs is a call to print function (non compliant)
  print ("hello") + " how are you" # OK (avoid Python2 FP)
  something.print ("hello") + " how are you" # Noncompliant

def pointless_statement_avoid_unused_import():
  math # OK, only 1 reference, might be done to suppress unused import issues
  numpy # Noncompliant
  numpy # Noncompliant
  pass

def no_issue_within_contextlib_suppress(a):
  from contextlib import suppress
  with suppress(TypeError):
    a + ''  # OK
    return a

from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
def airflow_ignore_context():
    with DAG("my-dag"):
        ping = HttpOperator(endpoint="http://example.com/update/")
        ping

def airflow_ignore_context_false_negative():
    with DAG("my-dag"):
        ping = HttpOperator(endpoint="http://example.com/update/")
        download = HttpOperator(endpoint="http://example.com/download/")
        upload = HttpOperator(endpoint="http://example.com/upload/")
        ping # Noncompliant
        download >> upload

from airflow.decorators import dag

@dag
def airflow_decorator_dag():
    ping = HttpOperator(endpoint="http://example.com/update/")
    ping

@dag
def airflow_decorator_multiple_statements():
    ping = HttpOperator(endpoint="http://example.com/update/")
    download = HttpOperator(endpoint="http://example.com/download/")
    upload = HttpOperator(endpoint="http://example.com/upload/")
    download >> upload
    ping # Noncompliant
    ping >> upload

def airflow_nested_with():
    with DAG("my-dag"):
        with open("some_file.txt") as file:
            ping = HttpOperator(endpoint="http://example.com/update/")
            download = HttpOperator(endpoint="http://example.com/download/")
            upload = HttpOperator(endpoint="http://example.com/upload/")
            ping # Noncompliant
            download >> upload

@dag
def airflow_decorator_no_dag():
    x = 3
    x # Noncompliant
    return x

def airflow_ignore_context_not_operator():
    with DAG("my-dag"):
        some_var = True
        some_other_var = True
        some_var # Noncompliant
        some_other_var
