import builtins


def notBasicBuiltinExceptions():
  TypeError() # Noncompliant
# ^^^^^^^^^^^
  Exception("") # Noncompliant {{Raise this exception or remove this useless statement.}}
# ^^^^^^^^^^^^^

# custom class inside same function
def notThrowingCustomException():
  class Custom(TypeError):
    pass

  Custom() # Noncompliant {{Raise this exception or remove this useless statement.}}
# ^^^^^^^^

# custom class outside of any function
class C1(TypeError):
          pass

def customException():
    C1()  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^

def coverage():
  SomethingUnknown()
  SomethingUnknown

  class C2(C3):
    pass

  C2()

def falseNegatives():

  # Deeper inheritance hierarchies currently don't work.
  class C1(TypeError):
    pass

  class C2(C1):
    pass

  C2() # FN. Doesn't report anything, because the inheritance hierarchy of exceptions is currently a linear list,
       # and the isOrExtends is not used.

  # Binders don't bind.
  e = TypeError() # FN. The invocation of the `TypeError` constructor is not a statement, it's an expression.


# special treatment of WindowsError-s
try:
    WindowsError  # Ok. This will raise an exception if the program doesn't run on Windows
    WindowsError()
except NameError:
    pass

# rest mutably borrowed from `expected-issues`
class CustomException(TypeError):
  pass

def InstantiatedBuiltinExceptions():
    BaseException()  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^^^
    Exception()  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^^^
    ValueError()  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^
    CustomException()  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^^^^^

    BaseException  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^
    Exception  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^
    ValueError  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^^
    CustomException  # Noncompliant {{Raise this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^^^


def compliant(param, func):
    exception = ValueError()
    exception

    def exception_factory() -> ValueError:
        return ValueError()

    exception_factory()

    class CustomException(TypeError):
        def method(self):
            self

    try:
        WindowsError
    except NameError:
        WindowsError = OSError

    try:
        ModuleNotFoundError
    except NameError:
        ModuleNotFoundError = ImportError

    try:
        BaseException
    except NameError:
        BaseException = Exception

    try:
        WindowsError
    except (NameError, ImportError):
        WindowsError = OSError

    try:
        WindowsError
    except builtins.NameError:
        WindowsError = OSError

    try:
        WindowsError
    except:
        WindowsError = OSError

    try:
        WindowsError
    except Exception:
        WindowsError = OSError

    try:
        WindowsError
    except BaseException:
        WindowsError = OSError

    try:
        WindowsError
    except NameError:
        if param:
            WindowsError = OSError

    lambda: ValueError() if param else None
    func(ValueError())
    if param == 1:
        raise ValueError() # added constructor invocation (previously no round parens)
    elif param == 2:
        raise ValueError()
    return ValueError()

def gen():
    yield ValueError()


def nested_function_is_not_an_availability_probe():
    try:
        def helper():
            ValueError  # Noncompliant {{Raise this exception or remove this useless statement.}}
        helper()
    except NameError:
        ValueError = Exception


def availability_probe_must_catch_name_error():
    try:
        WindowsError  # Noncompliant {{Raise this exception or remove this useless statement.}}
    except TypeError:
        WindowsError = OSError

    try:
        WindowsError  # Noncompliant {{Raise this exception or remove this useless statement.}}
    except custom.NameError:
        WindowsError = OSError


def fallback_assignments_in_nested_scopes_are_ignored():
    try:
        ValueError  # Noncompliant {{Raise this exception or remove this useless statement.}}
    except NameError:
        def helper():
            ValueError = Exception

    try:
        ValueError  # Noncompliant {{Raise this exception or remove this useless statement.}}
    except NameError:
        class Fallback:
            ValueError = Exception


def comprehension_scoped_fallback_is_not_an_availability_probe():
    try:
        ValueError  # Noncompliant {{Raise this exception or remove this useless statement.}}
    except NameError:
        [ValueError for ValueError in ()]


def import_fallback_is_an_availability_probe():
    try:
        ConnectionError
    except NameError:
        from socket import error as ConnectionError


def annotated_fallback_is_an_availability_probe():
    try:
        PermissionError
    except NameError:
        PermissionError: type = OSError


try:
    ModuleNotFoundError
except ImportError:
    ModuleNotFoundError = ImportError
