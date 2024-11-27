def function_with_local():
    a = 11
    a += 1
    a.x = 1
    foo(a)
    t2: str = "abc"
    b.a = 1
    foo().x *= 1
    foo().x : int = 1

global_x = 10

def function_with_free_variable():
    foo(global_x) # x is a free variable, not in local variable


def function_with_rebound_variable():
    foo(global_x) # x is bound, it's in local variable
    global_x = 1

def simple_parameter(a):
    pass

def multiple_assignment():
    x, y = (1, 2)

def tuple_assignment():
    (x, y) = (1, 2)

global_var = 1

def function_with_global_var():
    global global_var
    global_var = 10

def function_with_nonlocal_var():
    nonlocal non_local_var
    non_local_var = 10

def function_with_nested_nonlocal_var():
    x = 10
    def innerFn():
        nonlocal x
        x = 4

def function_with_lambdas():
    x = 42
    foo((lambda x: x*x))
    y = 42
    foo((lambda z: z*y))

def function_with_loops():
    for (x, y) in [1,2,3]:
        pass

    for [x1, y1] in [1,2,3]:
        pass


def function_with_comprehension():
    [2 for a in range(3)]

def func_wrapping_class():
    class A:
        myParam = 2

def var_with_usages_in_decorator():
    x = 10
    y = 10
    z = 10
    @x
    def foo(): pass
    @y.bar
    def foo(): pass
    @z.bar()
    def foo(): pass

def function_with_unused_import():
    import mod1
    import mod2 as aliased_mod2
    from mod3 import x
    from mod3 import y as z

def binding_usages(param):
    pass

def func_with_tuple_param((a, (b, c)), d):
  pass


def func_with_star_param(a, *, d):
  pass

class a:
  def method_with_star_param(*, d):
    pass

def print_var():
  print = 42

def symbols_in_comp():
  [x+y+z for (x, (y, z)) in [(1,(2,3)), (3,(4,5))]]

def for_comp_with_no_name_var():
  # test for comp that are not names nor tuple
  [x for fun() in [1,2]]

def scope_of_comprehension(x):
  [x+1 for x in [1,2]]
  foo(x)

def comprehension_reusing_name(a):
  {a:1 for a in a.foo}

def ref_in_interpolated(p1):
  fun(f"fun{p1}")

def fn_inside_comprehension_same_name():
    def fn():
        return [fn() for fn in [lambda: 1, lambda: 2]]

def exception_instance():
    try:
        pass
    except Exception as e1:
        pass

    try:
        pass
    except Exception as (e2,e3):
        pass

    try:
        pass
    except Exception as [e4,e5]:
        pass

    try:
        pass
    except Exception as (e6):
        pass

def with_instance():
    with open() as file1:
        pass

    with open() as (file2, file3):
        pass

def unpacking():
    *foo = f()

def using_builtin_symbol():
    print(42)

def keyword_usage():
    x = 42
    f(x = 43) # keyword x is not the same symbol of x defined as local var

def comprehension_vars():
    [42 for [a] in range(3)]

def parameter_default_value():
    foo = 42
    def func(x = foo):
        foo = 43

def assignment_expression():
    if (b:=foo()) != 42:
        bar(b)

def assignment_expression_in_generator():
    if any((last := i) for i in range(5)):
        something(last)

def assignment_expression_in_list_comprehension():
    if [last := i for i in range(5)]:
        something(last)

def assignment_expression_in_set_comprehension():
    if {(last := i) for i in range(5)}:
        something(last)

def assignment_expression_in_dict_comprehension():
    if {'test': (last := i) for i in range(5)}:
        something(last)

def importing_stdlib():
  import os.path
  os.path.realpath("")


def importing_submodule():
    import werkzeug.datastructures
    werkzeug.datastructures.Headers()


def importing_submodule_as():
    import werkzeug.datastructures as wd
    wd.Headers()

def importing_submodule_after_parent():
    import werkzeug
    import werkzeug.datastructures
    werkzeug.datastructures.Headers()


def importing_submodule_after_parent_nested():
    import werkzeug
    import werkzeug.datastructures
    import werkzeug.datastructures.csp
    werkzeug.datastructures.csp.ContentSecurityPolicy()


def importing_parent_after_submodule():
    import werkzeug.datastructures.csp
    import werkzeug
    werkzeug.datastructures.csp.ContentSecurityPolicy()


def importing_parent_after_submodule_2():
    import werkzeug.datastructures.csp
    import werkzeug.datastructures
    import werkzeug
    werkzeug.datastructures.csp.ContentSecurityPolicy()
    werkzeug.datastructures.Headers()


def importing_submodule_twice():
    import werkzeug.datastructures
    import werkzeug.datastructures
    werkzeug.datastructures.Headers()


def importing_unknown_submodule():
    import werkzeug.datastructures.unknown
    import werkzeug.datastructures.unknown
    werkzeug.datastructures.Headers()

def type_params[T: str]():
    a : T = "abc"
    return a

def type_alias():
    type M = str
    return list[M]
