def bare_raise_in_finally(param):
  result = 0
  try:
    print("foo")
  except:
    if param:
      raise ValueError()
  finally:
    if param:
      raise  # Noncompliant {{Refactor this code so that any active exception raises naturally.}}
#     ^^^^^
    else:
      result = 1
      class SomeClass():
        raise # OK (handled by S5747)
      def func():
        raise # OK (handled by S5747)


  return result


def bare_raise_in_except(param):
  result = 0
  try:
    print("foo")
  except:
    if param:
        raise
  finally:
    if not param:
      result = 1
  return result

def finally_in_except():
  try:
    raise TypeError()
  except TypeError:
    try:
      raise OSError()
    finally:
      raise

def finally_in_nested_method():
  try:
    raise TypeError()
  except TypeError:
    def nested():
      try:
        raise OSError()
      finally:
        raise # Noncompliant

def finally_in_nested_class():
  try:
    raise TypeError()
  except TypeError:
    class Nested:
      try:
        raise OSError()
      finally:
        raise # Noncompliant

class RaiseInExitMethod(object):
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, traceback):
    try:
      print("foo")
    finally:
      # No issue when `raise` is in __exit__ method. (S5706 handles it)
      raise

def raise_exception():
  try:
    pass
  except:
    pass
  finally:
    raise BaseException() # OK, not a bare raise

raise # OK, not in finally
