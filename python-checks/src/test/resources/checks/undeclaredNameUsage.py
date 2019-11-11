def f():
    print(a) # Noncompliant {{a is not defined. Change its name or define it before using it}}
#         ^

x = 10
def from_parent_scope():
    print(x) # OK

def use_global_x():
    global x
    print(x) # OK
    global y
    y = 42
    print(y)

def use_imported_name():
    from mod import fn # mod is OK, not a variable
    print(fn) # OK

print(foo = 10) # OK, keywords arguments are not variables


from mod import DecParam, Foo

@Foo(DecParam)
class A:
    pass

def types() -> float:
    x: int = 42 # type def OK

def comprehension_scope():
    [y for i in range(1,10)]
    # FP in Python 2 because of difference in scoping
    print(i) # Noncompliant

def used_before_definition():
    print(y) # Noncompliant
    y = 10

def maybe_used_before_definition(p):
    if p:
        y = 10
    print(y) # OK, might be initialized

def fun_def():
    func()  # Noncompliant
    def func():
        pass

def fun_def_ok():
    def func():
        pass
    func()

def class_def():
    MyClass()  # Noncompliant
    class MyClass:
        pass

def class_def_ok():
    class MyClass:
        pass
    MyClass()

def loop():
    for x in [0, 1, 2]: pass
    print(x)  # Ok. variables are accessible after a loop

def conditional():
    if False:
        condition_var = 0
    else:
        print(condition_var)  # Noncompliant {{condition_var is used before it is defined. Move the definition before.}}

def except_instance():
    try:
        pass
    except Exception as e:
        print(e)
    print(e)

def recursive_fn():
    def inner(): inner()
    inner()

def dead_code():
    x = 10
    print(x)
    return
    print(x) # OK

def class_def():
    class A(): pass
    a = A()

def lambda_def():
    inner = lambda : y + 1
    y = 10

myglobal = 42

def shadowing_ok():
    myglobal = 21  # This variable is local
    print(myglobal)  # Ok

def shadowing_nok():
    # (python will fail with "local variable 'myglobal' referenced before assignment")
    print(myglobal)  # Noncompliant
    myglobal = 42  # the variable is assigned, which makes it local

def try_except(condition):
    try:
        if condition:
            raise Exception('')
        res = 42
    finally:
        print(res)  # FN

def with_stmt():
    with A() as a:
        print(a) # OK

def declared_in_while(p):
    while p:
        if p == 10:
            var = 42
    print(var)

def compound_assignment():
    a += 1 # FN

def with_usages():
    with f() as a, a() as b: # OK
        pass
