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
    print(y) # OK, handled by S3827
    y = 10

def loop():
    for x in [0, 1, 2]: pass
    print(x)  # Ok. variables are accessible after a loop

def recursive_fn():
    def inner(): inner()
    inner()

def lambda_def():
    inner = lambda : y + 1
    y = 10

def default_parameter():
    foo = 42
    def f(param = foo): # OK
        foo = 'hello'

def one_issue_per_unresolved_name():
    print(xxx) # Noncompliant {{xxx is not defined. Change its name or define it before using it}}
  #       ^^^
    print(xxx) # OK, don't raise the same issue multiple times
  #       ^^^<

def one_issue_per_unresolved_name_2():
    print(xxx) # OK, don't raise the same issue multiple times
  #       ^^^<

def one_issue_per_unresolved_name_3():
    print(yyy) # Noncompliant
    def inner():
        pass
    print(yyy) # OK, don't raise the same issue multiple times

def foo(param):
    return 42

print(f'{foo(param=3)}')  # OK, param is a keyword argument

print(f'{foo(param)}')  # Noncompliant
#            ^^^^^

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

_some_implicit_global_vars # ok we exclude variables starting with `_`
__some_implicit_global_vars__ # ok


def python_3_10(value):
    match value:
        case remaining:
            print(remaining)

class InnerClassFp:
    class MyInnerClass:
        pass

    field : MyInnerClass # Ok

    class AnotherInnerClass:
        pass

    def __init__(self, my_inner: MyInnerClass):  # Ok
        # Within the scope of the function body, we should use `self.AnotherInnerClass`
        v1: MyInnerClass # Noncompliant
           #^^^^^^^^^^^^
        v2: self.MyInnerClass # Ok

        def even_inner(x: AnotherInnerClass): # Noncompliant
                         #^^^^^^^^^^^^^^^^^
            pass

    def ret(self) -> MyInnerClass: # Ok
        pass

    def class_in_function(self):
        class ClassInFunction:
            pass

        def nested_function() -> ClassInFunction:
            pass


def type_aliases():
    type A = B
    type B = getType()

    def getType():
        return str

    class C[A]():
        ...

def undefined_generic_type[T: T](a: T) -> T:
    x: T = a
    a.foo()

class TypeVarTest[T]:
    def from_class_def(self):
        a: list[T] = ...

    def unknow_generic(self):
        a: list[X] = ... # Noncompliant

# FP with python 2 syntax for exception handling
try:
  ""
except OSError, why: # Noncompliant
      pass
