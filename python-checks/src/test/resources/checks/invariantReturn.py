def foo(*argv):
    pass

def f_return_nothing_are_ignored(x):
    if x:
        return
    else:
        return

def f_return_none_are_ignored(x):
    if x:
        return None
    else:
        return None

def f_return_none_mixed_with_others(x):
    if x > 1:
        return None
    elif x > 3:
        return
    elif x > 5:
        return 2
    elif x > 7:
        return 2
    foo("A")

def f_one_return_is_excluded():
    foo("x")
    return 1

def f_same_string(x): # Noncompliant {{Refactor this method to not always return the same value.}}
#   ^^^^^^^^^^^^^
    if x:
        return "ab"
#       ^^^^^^^^^^^<
    else:
        return "ab"
#       ^^^^^^^^^^^<

def f_same_number(x): # Noncompliant
    if x:
        return 42
    return 42

def f_same_binary_and_unary_expression(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if x:
        return -(+5 * 2)
#       ^^^^^^^^^^^^^^^^<
    else:
        return -(+5 * 2)
#       ^^^^^^^^^^^^^^^^<

def f_different_binary_and_unary_expression(x):
    if x:
        return -(+5 * 2)
    else:
        return -(+5)

def f_different_number_in_if_else(x):
    if x:
        return 1
    else:
        return 2

def f_different_number_in_if(x):
    if x:
        return 1
    return 2

def f_no_return(x):
    pass

def f_constant_and_call_expression_are_different(x):
    if x:
        return 1
    foo()

def f_empty_string_are_the_same(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if x:
        return ""
#       ^^^^^^^^^<
    else:
        return ""
#       ^^^^^^^^^<

def f_different_binary_expression(x):
    if x:
        return "" + ""
    else:
        return "" + "b"

def f_same_binary_expression(x, a): # Noncompliant
    if x:
        return "" + a
    else:
        return "" + a

def f_return_in_except(): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^
    try:
        foo(2)
    except Error as e:
        return 1
#       ^^^^^^^^<
    return 1
#   ^^^^^^^^<

def f_different_return_in_except():
    try:
        return 1
    except:
        return 0
    return 1

def f_different_return_in_try():
    try:
        return 1
    except:
        return 0
    except:
        return 0
    return 0

def f_return_before_except(): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^
    try:
        return 0
#       ^^^^^^^^<
    except A:
        foo()
    finally:
        x = y
    return 0
#   ^^^^^^^^<

def f_ignored_return_when_finally_also_return(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if x:
        try:
            return "ignored because of finally"
        except:
            return "ignored because of finally"
        finally:
            return 0
#           ^^^^^^^^<
    else:
        return 0
#       ^^^^^^^^<

def f_return_in_for_loop(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    for i in x:
        return 0
#       ^^^^^^^^<
    return 0
#   ^^^^^^^^<

def f_return_in_for_loop_with_else(x): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    for i in x:
        return 0
#       ^^^^^^^^<
    for i in x:
        if i == 2:
            break
        else:
            return 0
#           ^^^^^^^^<
    else:
        return 0
#       ^^^^^^^^<
    if x:
        try:
            return 0
#           ^^^^^^^^<
        except Error as e:
            return 0
#           ^^^^^^^^<
        except:
            return 0
#           ^^^^^^^^<
    elif x == 2:
        return 0
#       ^^^^^^^^<
    else:
        return 0
#       ^^^^^^^^<

def f_return_boolean_true(x): # Noncompliant
    if x:
        return True
    return True

def f_return_boolean_false(x): # Noncompliant
    if x:
        return False
    return False

def f_return_boolean_false_true(x):
    if x:
        return False
    return True

def f_return_boolean_true_false(x):
    if x:
        return True
    return False

def f_different_return_list(x):
    if x:
        return 1, 2
    return 1, 3

def f_return_list_with_different_size(x):
    if x:
        return 1, 2, 3
    return 1, 2

def f_same_return_list(x): # Noncompliant
    if x:
        return 1, 2
    return 1, 2

def f_binary_expression_mixing_constants_and_identifier(x):
    if x:
        return 1 + a
    return a + 1

def f_same_parameter_binding(x, a): # Noncompliant
    if x:
        return a
    else:
        return a

def f_same_assignment_binding(x): # Noncompliant
    a = 2
    if x:
        return a
    else:
        return a

def f_ignore_previous_assignment_binding(x, a): # Noncompliant
    a = 2
    f26(a)
    a = 3
    if x:
        return a
    else:
        return a

def f_several_previous_assignment(x):
    a = 2
    if x:
      a = 3
    if x:
        return a
    else:
        return a

def f_previous_assignment_or_parameter(x, a):
    if x:
      a = 3
    if x:
        return a
    else:
        return a

def f_previous_assignment_in_a_loop(x, i):
    a = 2
    while i < 0:
        a = 3
        i += 1
    if x:
        return a
    else:
        return a

def f_common_previous_modification(x, a): # Noncompliant
    if x:
        a = 3
    a += 1
    if x:
        return a
    else:
        return a

def f_same_list_assignment(x): # Noncompliant
    a, b = 1, 2
    if x:
        return a
    else:
        return a

def f_list_assignment(x):
    a, b = 1, 2
    if x:
        return a
    else:
        return b

def f_assignment_in_the_return_block(x, a):
    if x:
        a = 4
        return a
    else:
        return a

def f_subscript_assignment(x, a):
    if x:
        a['name'] = 4
        return a
    else:
        return a

def f_subscript_read_usage(x, a): # Noncompliant
    if x:
        x = a['name']
        return a
    else:
        return a

def f_nested_modification(x, y):
    a = 0
    if x:
        if y:
            a = 3
        return a
    else:
        return a

def f_modification_after_return_but_in_a_loop(x, y):
    a = 0
    for e in x:
        if y == e:
            return a
        a += 1
    return a

def f_conditional_return_in_a_loop(x, y): # Noncompliant
    a = 0
    for e in x:
        if y == e:
            return a
    return a

def f_for_when_it_changes_the_value(x, y):
    a = 0
    if x:
        return a
    for a in x:
        pass
    return a

def f_for_when_it_does_not_change_the_value(x, y): # Noncompliant
    a = 0
    if x:
        return a
    for i in a:
        pass
    return a

def f_while_does_not_change_the_value(x, y): # Noncompliant
    a = 0
    if x:
        return a
    while a:
        pass
    return a

def f_same_global_variable(x): # Noncompliant
    global a
    if x:
        return a
    else:
        return a

def f_different_global_variable(x):
    global a
    if x:
        return a
    else:
        a = 3
        return a

def f_function_call_could_modify_identifier(x):
    if x:
        return a
    else:
        foo(a)
        return a

def f_name_and_call_expression_are_different(x, a):
    if x:
        return a
    return a()

def f_symbol_used_in_binary_expression_and_right_hand_side_assignment_is_not_modified(x, a): # Noncompliant
    if x:
        return a
    b = a
    foo(a or not (a) and (a - 7) > (a + b))
    return a

def f_unary_expressions_different_operators(a, b):
    if a:
        return +b
    return -b

def f_same_qualified_identifier(a, b): # false-negative, limitation with qualified identifier
    if a:
        return b.x
    return b.x

def f_raise_is_not_a_return(a, b):
    if a:
        raise b
    return b

def f_function_exit_through_raise_should_be_ignored(a, b, c): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if a:
        return b
#       ^^^^^^^^<
    if c:
        raise
    return b
#   ^^^^^^^^<

def f_same_binding_through_multiple_paths(a): # Noncompliant
    d = 3
    try:
        if a:
            return d
    except TypeError as e:
        foo(e)
    except NameError as e:
        foo(e)
    return d

def f_a_lot_of_binding(a):
    d = 3
    try:
        if a:
            return d
        d = 4
    except TypeError as e:
        d = 5
    except NameError as e:
        d = 6
    return d

def with_assignment_expression(a, b): # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  if (b := 12) == a:
     return b
#    ^^^^^^^^<
  else:
     return b
#    ^^^^^^^^<

def no_fp_on_assignment_expression(a, b):
    if a:
        a = (b := 42)
        return b
    else:
        return b

def no_fp_implicit_return_none(a, b):
  if a:
    return 42
  elif b:
    return 42
  # implicitly returns "None"

def cannot_open_file(path):
  try:
    open(path)
  except FileNotFoundError as e:
    return True
  except IsADirectoryError as e:
    return True
 # implicitly returns "None"


def fp_try_statement_reachable_implicit_none_return(cond):  # Noncompliant
  try:
    if cond:
      return 42
  except FileNotFoundError:
    return 42
  # implicitly returns "None"

def unreachable_implicit_return_none(cond):  # Noncompliant
  try:
    return 42
  except Exception:
    return 42
  # Implicit "None" return is unreachable
