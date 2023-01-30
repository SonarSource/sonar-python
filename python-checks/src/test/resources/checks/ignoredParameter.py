def ignored_param(p): # Noncompliant
    p = 42
    print(p)

def reassignment_on_param_after_read(p):
    print(p)
    p = 42  # OK

def ignored_param_default(p = 10): # Noncompliant
#                         ^^^^^^
    p = 42
    print(p)

def invalid_cfg():
    continue

def only_write(param): # Noncompliant
#              ^^^^^
    param = 42

def unused(param): # OK
    pass

def dead_code(param):
    return
    param = 42

def tuple_param((param)): # Noncompliant
    param = 42

def param_can_be_falsy(p): # OK
    p = p or 42
    print(p)

def used_in_subfunction(p):
    def fn():
        nonlocal p
        print(p)
        p = 42
    fn()

def underscore_param(_): # OK
    _ = 42

def assignment_expression(p): # Noncompliant
  foo(p:=bar())

def assignment_expression_fn(a): # FN (first dict key computation overwrites "a" before it's read)
  dict = {'b' : (a:=3), 'c' : a}

def assignment_expression_no_fp(a):
  dict = {'b' : a, 'c' : (a:=3)} # OK, read before write

def match_statement_no_fp(value, param):
  match value:
    case param.CONST: param = 42
    case "other": ...
  value = 42  # OK

def match_statement_no_fn(value, param): # Noncompliant
  match value:
    case 1: ...
    case param: ...
  value = 42  # OK

def secondary_location(p): # Noncompliant {{Introduce a new variable or use its initial value before reassigning 'p'.}}
#                      ^ 1
    p = 42
#   ^^^^^^< 1 {{'p' is reassigned here.}}
    p = 43
    p = 44
    print(p)

def secondary_location_internal_blocks(a, b):  # Noncompliant {{Introduce a new variable or use its initial value before reassigning 'b'.}}
#                                         ^ 2
    while a():
        b = 42
#       ^^^^^^< 1 {{'b' is reassigned here.}}
        print(b)
    b = 24
#   ^^^^^^< 2 {{'b' is reassigned here.}}
    print(b)

def secondary_location_multiple_internal_blocks_2(a, b):  # Noncompliant {{Introduce a new variable or use its initial value before reassigning 'b'.}}
#                                                    ^ 3
    if a():
        b = 42
#       ^^^^^^< 1 {{'b' is reassigned here.}}
    else:
        b = 41
#       ^^^^^^< 2 {{'b' is reassigned here.}}
        print(b)
    b = 24
#   ^^^^^^< 3 {{'b' is reassigned here.}}
    print(b)

def ignored_param_reassign_in_loop(p, b): # Noncompliant {{Introduce a new variable or use its initial value before reassigning 'p'.}}
#                                  ^ 1
    for p in range(b):
#       ^< 1 {{'p' is reassigned here.}}
        print(p)
    p = 42
    print(p)
