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
