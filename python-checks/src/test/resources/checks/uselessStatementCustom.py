"""docstring are still ignored"""

"not a docstring" # Noncompliant
class Class:
  """docstring"""

  """not a docstring""" # Noncompliant

def report_on_string():
  """docstrings"""
  "this is not" # Noncompliant
  'neither this' # Noncompliant
  """this is not a docstring""" # Noncompliant

def binary_operators():
  'single quoted docstring'
  42 + 24 # + is ignored
  a << b # << is ignored
  a >> b # Noncompliant
  a | b  # Noncompliant
  + 42 # OK: to avoid FPs, ignored operators are ignored for both unary and binary expressions
  50 - 8 # Noncompliant
