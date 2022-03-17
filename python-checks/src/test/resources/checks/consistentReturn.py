# simple cases

def no_return_at_all(x):
  print(x)

def return_value_on_all_branches(x):
  if x > 0:
    return x
  else:
    return None

def return_without_value(x): # Noncompliant {{Refactor this function to use "return" consistently.}}
#   ^^^^^^^^^^^^^^^^^^^^
  if x < 0:
    return
#   ^^^^^^ < {{Return without value}}
  return x
# ^^^^^^^^ < {{Return with value}}

def all_returns_have_no_value(x):
  if x > 0:
    print(x)
    return
  print(0)
  return


# implicit return

def return_without_value_and_implicit_return(x):
  if x > 0:
    print(x)
    return
  print(0)

def implicit_return_on_expression_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
    return x
#   ^^^^^^^^ < {{Return with value}}
  print(x)
# ^^^^^^^^ < {{Implicit return without value}}


# implicit return and compound statements

def implicit_return_on_if_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
# ^^ < {{Implicit return without value if the condition is false}}
    return x
#   ^^^^^^^^ < {{Return with value}}

def implicit_return_on_for_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  for e in x:
# ^^^ <
    return x
#   ^^^^^^^^ <

def implicit_return_on_while_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  while condition(x):
# ^^^^^ <
    x = foo()
    if x > 0:
      return x
#     ^^^^^^^^ <

def while_true(x):
  while True:
    x = get_random_string(32)
    if valid(x):
      return x

def while_with_unknown_condition(x): # Noncompliant
  while condition(x):
    x = get_random_string(32)
    if valid(x):
      return x

def implicit_return_on_def_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
    return x
#   ^^^^^^^^ < {{Return with value}}
  def foo():
# ^^^ < {{Implicit return without value}}
    pass

def implicit_return_on_class_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x > 0:
    return x
#   ^^^^^^^^ < {{Return with value}}
  class Foo:
    def meth1():
#   ^^^ < {{Implicit return without value}}
      print("meth1")

def implicit_return_on_match_statement(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  match(x):
# ^^^^^ < {{Implicit return without value when no case matches}}
    case 0:
      return x
#     ^^^^^^^^ < {{Return with value}}

def match_statement_with_default(x): # Noncompliant
  match(x):
    case 0:
      return x
    case _:
      return 42

# raise / assert

def return_value_or_raise(x):
  if x > 0:
    return x
  raise "error"

def return_value_or_assert(x):
  if x > 0:
    return x
  assert False, "x should be > 0"

def no_secondary_location_on_raise_or_assert(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  if x == 0:
    raise "error"
  if x > 99:
    assert False, "unexpected"
  if x < 0:
    return
#   ^^^^^^ <
  return x
# ^^^^^^^^ <


# Limitations (CFG and others)

def bad_cfg(x):
  break

def try_except(x):
  try:
    return foo()
  except Exception:
    return 0

def try_finally(x):
  if x > 0:
    return 0
  try:
    return x
  finally:
    print(x)

def noncompliant_function_containing_try_in_nested_function(p): # Noncompliant
  def nested_function_with_try(x):
    try:
      return x
    finally:
      print(x)
  if p:
    return nested_function_with_try

def with_statement(x):
  with something(x):
    return foo(x)

def with_statement_inside_if(x): # Noncompliant
  if x > 0:
    with something(x):
      return foo(x)

def fp_on_undetected_raise(x): # Noncompliant
  if x > 0:
    return x
  raise_error()

def raise_error():
  raise "error"
