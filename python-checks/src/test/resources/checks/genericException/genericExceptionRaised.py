def bare_raise():
    raise

def base_exception():
    raise BaseException()  # Noncompliant {{Replace this generic exception class with a more specific one.}}
#         ^^^^^^^^^^^^^^^

def exception():
    raise Exception()  # Noncompliant

class MyException(Exception):
    pass

def other_exception():
    raise MyException()

class MyClass():
    pass

def not_an_exception():
    raise MyClass()

def string():
    raise "Hello"

def no_instantiation(p):
    if p == 0:
        raise BaseException # Noncompliant
    elif p == 1:
        raise MyException
    else:
        raise UnknownSomething

def variable(cond):
    if cond:
        e1 = Exception()
        raise e1 # Noncompliant
    else:
        e2 = MyException()
        raise e2

def python2_multiple_expressions(cond):
    if cond:
        raise 42, BaseException
    else:
        raise BaseException, 42 # Noncompliant


def no_issue_with_self_return_type():
    raise MyException().with_traceback("foo")

def raised_exception_is_the_parameter(exception: BaseException):
    raise exception

global_exception = BaseException()
def raise_global_exception():
    raise global_exception

def handle_exception(e: Exception):
    raise e

def intermediate_handle_exception(e: Exception):
    handle_exception(e)

def constructed_exceptions_passed_to_handle_exception_raise():
    handle_exception(Exception()) # Noncompliant
    handle_exception(BaseException()) # Noncompliant

def is_instance_do_not_raise(e):
    isInstance(e, Exception)
