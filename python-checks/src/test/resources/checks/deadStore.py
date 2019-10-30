def empty(a) : pass

a = 1

def global_scope_not_local():
    a = 42 # has the same name as the global a, but is in local scope
    print(a)

def global_scope():
    global a
    a = 21 # compliant, global scope is ignored
    print(a)

def nonlocal_scope():
    nonlocal a
    a = 21 # compliant, nonlocal scope is ignored
    print(a)

def params_are_ignored(x, y):
    x = 42
    y = 43
    print(x)
    print(y)

def simple_assignments():
    x = 42 # Noncompliant {{Remove this useless assignment to local variable 'x'.}}
#   ^^^^^^
    x = 3
    print(x)

def increment():
    x = 42
    print(x)
    x = x + 1 # both read and write
    y = 31
    y = y - 1

def minus():
    x = 42
    y = -x
    return x

def chain_assign():
    a = b = 0 # Noncompliant 2
#   ^^^^^^^^^
    a = foo()
    b = 42
    print(a, b)

def compound_assignment():
    z = 42
    z *= foo()
    print(z)


def tuple_assign():
    (a, b) = foo() # Noncompliant 2
    (a, b) = foo()
    print(a)
    print(b)

    (c, d) = foo()
    print(c)
    print(d)
    (c, d) = foo() # Noncompliant 2


def keyword_param():
    x = 42
    print(x)
    foo(x = 'hello') # OK

def simple_conditional():
    x = 10 # Noncompliant
    if p:
        x = 11
        print(x)

def used_inside_conditional():
    x = 10
    if p:
        print(x)
    else:
        x = 11 # Noncompliant

def used_after_conditional():
    x = 10 # Noncompliant
    if p:
        print()
    x = 11
    print(x)

def used_inside_conditional_ok():
    x = 3
    if p:
        print(x)
        x = 4
    else:
        x = 5
    print(x)

def used_after_conditional_ok():
    x = 6
    if p:
        x = 7
    print(x)

def used_inside_nested_if_ok():
    x = 42
    if p:
        print()
        if p2:
            print(x)

def used_condition_if():
    x = 42
    if x: pass
    x = 10
    print(x)

def loop_nok():
    x = 42 # Noncompliant
    while p:
        x = foo()
        print(x)

def loop_only_assignment():
    x = 42
    print(x)
    while p:
        x = foo() # Noncompliant

def foreach_nok():
    elem = 42
    print(elem)
    x = 42
    for elem in foo(): # Noncompliant
        print(x)
    print()

def loops_rw():
    x = 42
    while p:
        x = x + 1

def loops_ok():
    x = 42
    while x < 42:
        x += 41

def try_except():
    try:
        foo(x)
        x = 10
        x = 11 # OK
    except:
        print(x)

class A:
    def fun(self):
        print(self.a)
        self.a = 11 # OK

def used_conditionally_in_foreach():
    found = False
    for _ in elems:
        if p:
            found = True
    assert found

def used_conditionally_in_while():
    found = False
    while p:
        if p:
            found = True
    assert found

# invalid cfg for coverage
def invalid_cfg():
    continue

def dead_store_with_compound_assignment():
    x = 1
    x += 1
    print("a") # FN

def outer_fn():
    def inner_fn_1():
        print(inner_fn_2())
    def inner_fn_2(): # OK
        pass
    print("outer_fn")
