x = 42
y = True
z = 1.0

a = "a"
b = "b"

def f0(x, y):
    ...

f0(y, x) # Noncompliant
f0(x, y) # Compliant
f0(a, b) # Compliant
f0(a, x) # Noncompliant
f0(a, y) # Compliant
f0(x, b) # Compliant
f0(42, y) # Compliant
f0(42, x) # Noncompliant
f0(x, x) # Noncompliant

def f1(x, y, z):
    ...

f1(x, y, z) # Compliant
f1(y, z, x) # Noncompliant
f1(x, z, y) # Noncompliant
f1(y, x, z) # Noncompliant

f1.__call__(y, x, z) # Compliant

def scope():
    class C:
        def f2(self, x, y):
            ...

    c = C()
    self = C()
    c.f2(x, y) # Compliant
    c.f2(y, x) # Noncompliant
    C.f2(c, x, y) # Compliant
    C.f2(c, y, x) # Noncompliant
    C.f2(self, x, y) # Compliant
    C.f2(c, self, y) # Noncompliant
    c.f2(self, y) # Compliant: In cases like this where the `self` paramter is implicitly bound. A `self` argument likely refers to another
                  #            object, and it is highly likely that it was passed intentionally at another position.
    c.f2(self, x) # Noncompliant

    class D:
        def f3(this, self):
            ...

    d = D()
    self = D()
    this = D()
    d.f3(this) # Compliant
    d.f3(self) # Compliant
    D.f3(d, this) # Noncompliant
    D.f3(d, self) # Compliant
    D.f3(this, self) # Compliant
    D.f3(self, this) # Noncompliant

def f4(x):
    ...

f4(x) # Compliant
f4(y) # Compliant
f4(42) # Compliant

def f5(x, y, /):
    ...

f5(x, y) # Compliant
f5(y, x) # Noncompliant

def f6(x, y, /, z):
    ...

f6(x, y, z) # Compliant
f6(x, z, y) # Noncompliant
f6(x, y, z=x) # Compliant
f6(z, y, z=x) # Noncompliant
f6(z, y, z=z) # Noncompliant

def f7(x, y):
    ...

f7(x=x, y=x) # Compliant
f7(x=y, y=x) # Compliant: If a user explicitly specifies the parameter name, it is likely intentional
f7(x, y=y) # Compliant
f7(x, y=x) # Compliant
f7(y, y=y) # Noncompliant

def f8(x=10, y=42):
    ...

f8(x, y) # Compliant
f8(y, x) # Noncompliant

def f9(x, y=x):
    ...

f9(x, y) # Compliant
f9(y, x) # Noncompliant
f9(x, y=x) # Compliant

def f10(x, *, y, z):
    ...

f10(x, y=y, z=z) # Compliant
f10(y, y=y, z=z) # Noncompliant
f10(x, y=x, z=z) # Compliant

def f11(x, /, y, *, z):
    ...

f11(x, y, z=z) # Compliant
f11(x, y, z=y) # Compliant
f11(z, y, z=z) # Noncompliant
f11(x, x, z=z) # Noncompliant
f11(x, y=x, z=x) # Compliant

# Python 2 syntax
def f12(x, (y, z), a):
    ...

f12(x, (y, z), a) # Compliant
f12(x, (z, y), a) # FN: No support for python 2 tuple argument syntax
f12(a, (y, z), x) # FN

def f13(x, **kwargs):
    ...

f13(x) # Compliant
f13(x, y=x) # Compliant
f13(y, y=x) # Compliant

def f14(x, y, **kwargs):
    ...

f14(y, y, z=x) # Noncompliant
f14(x, y=x, z=x) # Compliant
f14(y, y=y, z=z) # Noncompliant

def f15(x, y, *args):
    ...

f15(x, y) # Compliant
f15(y, x) # Noncompliant
f15(x, y, x) # Compliant
f15(y, y, 10) # Noncompliant
f15(y, y, x) # Noncompliant

def f16(x, y, /, z, *args, a, b=10, **kwargs):
    ...

f16(x, y, z, a=a) # Compliant
f16(y, y, z, a=a) # Noncompliant
f16(x, y, y, a=a) # Noncompliant
f16(x, y, z=y, a=a) # Compliant
f16(z, y, z=y, a=a) # Noncompliant
f16(x, y, z, x, a=a, x=10) # Compliant
f16(y, y, z, x, a=a, x=10) # Noncompliant
f16(x, y, z, b, a=a, b=10) # Compliant

def f17(x17, y17, z17):
    ...

def scope17():
    x17 = [1]
    y17 = [2]
    z17 = [3]
    f17(x17, *y17, z17) # Compliant
    f17(x17, *x17, z17) # Compliant
    f17(y17, *y17, z17) # Noncompliant

    x17 = [2, 3]
    f17(x17, *x17) # Compliant
    z17 = [1, 2, 3]
    f17(*z17) # Compliant


def f18(x18, y18, z18):
    ...

def scope18():
    x18 = 1
    y18 = {'y18': 2, 'z18': 3}
    z18 = {'x18': 1, 'y18': 2, 'z18': 3}

    f18(x18, **y18) # Compliant
    f18(y18, **y18) # Compliant
    f18(**z18) # Compliant

    y18 = [2]
    z18 = {'z18': 3}
    f18(x18, *y18, **z18)

    y18 = {'z18': 3}
    z18 = [2]
    f18(x18, *z18, **y18) # Compliant
    f18(z18, *z18, **y18) # Compliant

# Example from RSPEC
def move_point(coord, speed):
    new_x = coord[0] + speed[0]
    new_y = coord[1] + speed[1]
    return (new_x, new_y)

coord = (3, 4)
speed = (1, 2)
move_point(speed, coord)  # Noncompliant
move_point(coord, speed)  # Compliant
