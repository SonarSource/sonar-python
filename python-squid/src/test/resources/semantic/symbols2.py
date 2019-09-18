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

def function_with_lambdas():
    x = 42
    foo((lambda x: x*x))
    y = 42
    foo((lambda z: z*y))

def function_with_loops():
    for x in [1,2,3]:
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
