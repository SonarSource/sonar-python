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
