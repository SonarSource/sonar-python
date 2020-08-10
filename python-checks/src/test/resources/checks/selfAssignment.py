import import1, x as import2
from module1 import (import3, x as import4)

x = 1
x = y
x = x # Noncompliant
# ^
x: int = x # Noncompliant
#      ^
x: int
x += x
x = c = c # Noncompliant
x = c = x # Noncompliant
y = y = y # Noncompliant
# Noncompliant@-1
x, y = x

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

try:
    s = table[s]
except KeyError:
    s = s # Noncompliant

a.x = a.x # Noncompliant
a[x] = a[x] # Noncompliant
a[x]: str = a[x] # Noncompliant
a[sideEffect()] = a[sideEffect()]

import1 = import1
import2 = import2
import3 = import3
import4 = import4
import6 = import1
import1 : type = import1
a : type = import1


if (python2):
	unichr = unichr # ok, builtin
	buffer = buffer # ok, builtin

def assignment_expression():
  a = 42
  if (a:=a): # Noncompliant
    pass
  if (b:=foo()):
    b = b # Noncompliant

def unpacking():
  x = [1]
  x, = x  # OK
  y, = x = x # Noncompliant
#        ^
