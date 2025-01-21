a = 3
b = foo()
F"foo{b}"
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
[x+1 for x in a]
{key:1 for key in a}
f"answer is \
{b}"

u = 42
nested = f'some: {f"nested interpolation：{u}"}'

nested = f'some: {f"nested interpolation：\
{u}"}'

f"symbol created and used inside interpolation: {[len(x) for x in []]}"

v = 42
f'use of equal specifier here {v=}, here {v=} and also {f"somewhere: \
        around here {v=}"}'

def nested_f_string_using_symbol():
    foo = 42
    f'{bar(f"{foo}")}'
