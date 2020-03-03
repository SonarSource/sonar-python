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
