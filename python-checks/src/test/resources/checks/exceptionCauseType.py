class NotAnException:
    pass

class MyException(Exception):
    pass

def raise_without_from():
    raise TypeError()

def raise_from_exception():
    raise TypeError() from MyException()

def raise_from_string():
    raise TypeError() from "just a string" # Noncompliant

def raise_from_non_exception():
    raise TypeError() from NotAnException() # Noncompliant

def raise_from_none():
    raise TypeError() from None

def raise_from_unknown(e):
    raise TypeError() from e

def assign_cause_to_exception(e):
    e.__cause__ = MyException()

def assign_cause_to_string(e):
    e.__cause__ = "just a string" # Noncompliant

def assign_cause_to_non_exception(e):
    e.__cause__ = NotAnException() # Noncompliant

def assign_cause_to_none(e):
    e.__cause__ = None

def assign_cause_to_unknown(e):
    e.__cause__ = Unknown()

def other_assignments(e):
    e.__eq__ = NotAnException()
    e = NotAnException()
    e.__cause__ = f = NotAnException() # Noncompliant

def type_given_by_except_clause():
    try:
        foo()
    except MyException as e:
        raise MyException() from e
    except NotAnException as e:
        raise MyException() from e # FN
    except (ValueError, TypeError) as e:
        raise MyException() from e
    except (NotAnException, str) as e:
        raise MyException() from e # FN


def except_from_exception_type():
    try:
        foo()
    except ValueError:
        raise ValueError("Caught some value error") from ValueError

    class SomeClass:
        ...
    try:
        foo()
    except ValueError:
        raise ValueError("Caught some value error") from SomeClass  # FN SONARPY-1666


def reassigned_exception():
    my_exception = None
    my_exception = ZeroDivisionError
    try:
        ...
    except my_exception:
        raise my_exception("Caught some value error") from my_exception
