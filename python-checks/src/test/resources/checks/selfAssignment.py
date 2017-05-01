x = 1
x = y
x = x # Noncompliant
x += x

def f():
    x = x # Noncompliant

class C:
    x = x

    def f():
        x = x # Noncompliant

try:
    from hashlib import sha1
    sha1 = sha1
except ImportError:
    from sha import new as sha1
