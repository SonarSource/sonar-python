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
        pass;

def function_with_comprehension():
    [2 for a in range(3)]

def func_wrapping_class():
    class A:
        myParam = 2

def var_with_usages_in_decorator():
    x = 10
    @x
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
