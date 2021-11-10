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

def match_statement_fp(value, param): # Noncompliant
  match value:
    case param.CONST: param = 42 # FP here: b.CONST should be a reading usage of b
    case "other": ...
  value = 42  # OK

def match_statement_fn(value, param):
  match value:
    case 1: ...
    case param: ...  # FN, c is overridden without having been read
  value = 42  # OK
