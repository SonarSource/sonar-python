def duplicate_except():
  try:
    raise TypeError()
  except TypeError as e:
#        ^^^^^^^^^> {{Exceptions will be caught here.}}
    print(e)
  except TypeError as e:  # Noncompliant {{Catch this exception only once; it is already handled by a previous except clause.}}
#        ^^^^^^^^^
    print("Never executed")
  except TypeError as e:  # Noncompliant {{Catch this exception only once; it is already handled by a previous except clause.}}
    print("Never executed")

def within_tuple():
    try:
      raise ModuleNotFoundError()
     #x 2
    except (ImportError, OSError) as e:
#           ^^^^^^^^^^^>
        print(e)
    except (ModuleNotFoundError, TypeError) as e: # Noncompliant {{Catch this exception only once; it is already handled by a previous except clause.}}
#           ^^^^^^^^^^^^^^^^^^^
        print(e)

def bare_except():
  try:
    raise ValueError()
  except BaseException:
#        ^^^^^^^^^^^^^> {{Exceptions will be caught here.}}
    pass
  except: # Noncompliant {{Merge this bare "except:" with the "BaseException" one.}}
# ^^^^^^
    pass
  try:
    raise ValueError()
  except: # OK
    pass

class MyException(BaseException):
  pass

class MyChildException(MyException):
  pass

def custom_exceptions():
  try:
    raise MyChildException()
  except MyException as e:
    pass
  except MyChildException as e: # Noncompliant
    pass

def exception_class_assigned_to_variable():
  a = UnicodeDecodeError
  try:
    raise a
  except UnicodeError:
    pass
  except a: # FN, type of a is "class", missing link with corresponding class symbol
    pass

def unknown_symbols():
  something = get_some_exception_class()
  try:
    raise SomeException()
  except UnknownException:
    pass
  except (AnotherUnknownError, UnknownError):
    pass
  except something:
    pass
  except a or b: # will be handled by S5714
    pass

def fn_Exception_not_a_super_class():
  try:
    raise UnicodeDecodeError()
  except Exception as e:
    print(e)
  except ValueError as e:  # FN as "Exception" super class is missing
#        ^^^^^^^^^^>
    print("Never executed")
  except UnicodeError as e:  # Noncompliant
#        ^^^^^^^^^^^^
    print("Never executed")

def oserror_hierarchy():
    try:
        raise FileExistsError()
    except (OSError, RuntimeError) as e:  # Secondary x2
    #FileExistsError is a subclass of OSError
        print(e)
    except FileExistsError as e:  # Noncompliant
        print("Never executed")
    except FileNotFoundError as e:  # Noncompliant
        print("Never executed")
