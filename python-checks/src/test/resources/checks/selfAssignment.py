import import1, x as import2
from module1 import (import3, x as import4)

x = 1
x = y
x = x # Noncompliant
# ^
x: int = x # Noncompliant
#      ^
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

a.x = a.x # Noncompliant
a[x] = a[x] # Noncompliant
a[x]: str = a[x] # Noncompliant
a[sideEffect()] = a[sideEffect()]

import1 = import1
import2 = import2
import3 = import3
import4 = import4


if (python2):
	unichr = unichr # ok, builtin
