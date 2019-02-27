import myModuleName
import original as alias
from myModuleName import f

a = 1
a = 2
b = 1
t1: int = 2

def function_with_local():
    a = 11
    a.x = 1
    foo(a)
    t2: str = "abc"

def function_with_global():
    global a, t1
    a = 11
    c = 11
    t1: int = 3
    t3: int = 3

def nesting1():
    a = 11
    def nesting2():
        a = 21
        def nesting3():
            def nesting4():
                global a
                a = 41
                def function_with_nonlocal():
                    nonlocal a
                    a = 51

def compound_assignment():
    a += 1

def simple_parameter(a):
    pass

def list_parameter(((a, (b)))):
    pass

def unknown_global():
    global unknown
    unknown = 1
    foo(unknown)

def dotted_name():
    a = 1
    @a.x
    def decorated():
        pass

class C:
    a = a
    b = a
    c: int = b
    loaded = property(lambda self: self._loaded)

def function_with_lambdas():
    print([(lambda unread_lambda_param: 2)(i) for i in range(10)])
    x = 42
    print([(lambda x: x*x)(i) for i in range(10)])
    y = 42
    print([(lambda x: x*x + y)(i) for i in range(10)])
    print([(lambda: y)(i) for i in range(10)])
    {y**2 for a in range(3) if lambda x: x > 1 and y > 1}

@abc(k for k in range(4))
def function_with_loops():
    {i for i in range(3) if i < 2}
    [i for i in range(3) if i < 2 and i > 1]
    for j in [0, 1, 2]:
        do_something(j)
    for a, b in mylist:
        do_something(a)

def module_name(params):
    myModuleName.run(params)
    myModuleName.eval(params)
    f(params)
    alias.foo()

def calling_same_function_multiple_times(params):
    myModuleName.bar(params)
    myModuleName.bar(params)
    myModuleName.f(params)
    f(params)
