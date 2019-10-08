a = 3
b = foo()

def function_with_local():
    a = 11
    a += 1
    a.x = 1
    foo(a)
    t2: str = "abc"
    b.a = 1
    foo().x *= 1
    foo().x : int = 1
    toto(a)

class clazz:
    field = "a"

    def __init__(self):
        self.a = "abc"

    def some_func(self):
        self.a = "u"
        self.field = "b"

    def other_func(self):
        self.a = "c"

l = lambda z : z*z
