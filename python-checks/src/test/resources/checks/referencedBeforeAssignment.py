def f():
    print(a) # OK, handled by S5953

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

def conditional():
    if False:
        condition_var = 0
#       ^^^^^^^^^^^^^>
    else:
        print(condition_var)  # Noncompliant {{condition_var is used before it is defined. Move the definition before.}}
#             ^^^^^^^^^^^^^

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

def default_parameter():
    foo = 42
    def f(param = foo): # OK
        foo = 'hello'

def comprehension(list):
    [x1 for [x1] in list]

def declaration_in_try_with_break():
    for _ in range(1000):
        try:
            res = 42
            break
        except Exception:
            raise TypeError()
    return res # OK

def one_issue_per_symbol():
  print(xxx) # Noncompliant {{xxx is used before it is defined. Move the definition before.}}
#       ^^^
  print(xxx) # OK, don't raise the same issue multiple times
#       ^^^<
  xxx = "hello"
# ^^^<

def one_issue_per_symbol_2():
  print(zzz) # Noncompliant
#       ^^^
  def inner():
    print(zzz)
#         ^^^<
  zzz = "hello"
# ^^^<


def decorator(param):
    pass

class A:
    _ATTR = 42
    @decorator(_ATTR)  # OK
    def foo(self):
        print("foo")

class A:
    _ATTR = 42
    @decorator(_ATTR)  # OK
    class Foo: pass

def test_print_list():
    f"{ {element for element in [1, 2]} }" # OK

global GLOB  # OK
GLOB = 42

def use_glob():
    print(GLOB)

def use_glob2():
    print(GLOB2)  # OK

global GLOB2
GLOB2 = 42

def use_glob3():
  global x
  print(x) # OK
  x = 42
x = 24

global GLOB3
print(GLOB3) # FN

def assignment_expression():
  a = 41
  dict = {(s := a + 1) : s} # OK

def assignment_expression_fn():
  b = 41
  dict = {k: (k := b + 1)} # FN, key is evaluated first and value second

def match_statement_no_fp(value):
  match value:
    case x:
      ...
  x = 42

def match_statement_no_fp_reassignment(value):
  match value:
    case x:  # OK, though should be raised by S1854 (dead store)
      x = 42

# To be fixed in https://sonarsource.atlassian.net/browse/SONARPY-1524 ticket
def type_aliases_statement_fp_reference():
    type A = B # OK
    type B = getType() # OK

    def getType():
        return str

    class C[A]():
        ...
