def f():
    print(a) # Noncompliant {{Change its name or define it before using it}}
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
