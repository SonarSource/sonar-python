def foo():
  raise # Noncompliant {{Remove this "raise" statement or move it inside an "except" block.}}
# ^^^^^
  raise Error()

def __exit__():
  raise

def bar():
  try:
    ...
  except:
    raise
    def fn():
      raise # Noncompliant
  finally:
    raise

def function_called_inside_except():
  def inner():
    raise
  try:
    ...
  except:
    inner()

def function_not_called_inside_except():
  def inner():
    raise # Noncompliant
  inner()

raise # Noncompliant
