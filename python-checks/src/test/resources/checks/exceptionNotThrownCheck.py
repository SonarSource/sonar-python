def notBasicBuiltinExceptions():
  TypeError() # Noncompliant
# ^^^^^^^^^^^
  Exception("") # Noncompliant {{Throw this exception or remove this useless statement.}}
# ^^^^^^^^^^^^^

# custom class inside same function
def notThrowingCustomException():
  class Custom(TypeError):
    pass

  Custom() # Noncompliant {{Throw this exception or remove this useless statement.}}
# ^^^^^^^^

# custom class outside of any function
class C1(TypeError):
          pass

def customException():
    C1()  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^

def coverage():
  SomethingUnknown()
  SomethingUnknown

  class C2(C3):
    pass

  C2()

# rest mutably borrowed from `expected-issues`
class CustomException(TypeError):
  pass

def InstantiatedBuiltinExceptions():
    BaseException()  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^^^
    Exception()  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^^^
    ValueError()  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^
    CustomException()  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^^^^^

    BaseException  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^
    Exception  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^
    ValueError  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^^
    CustomException  # Noncompliant {{Throw this exception or remove this useless statement.}}
#   ^^^^^^^^^^^^^^^


def compliant(param, func):
    lambda: ValueError() if param else None
    func(ValueError())
    if param == 1:
        raise ValueError() # added constructor invocation (previously no round parens)
    elif param == 2:
        raise ValueError()
    return ValueError()

def gen():
    yield ValueError()
