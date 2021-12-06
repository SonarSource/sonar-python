import collections
from math import acos
from keyword import iskeyword
import pwd
from emoji import emojize
import platform

class A:
    pass

class B:
    pass

class ClassWithEq:
    def __eq__(self, other):
        pass

class ClassWithNe:
    def __ne__(self, other):
        pass

def f():
    if A() == A(): pass
    if A() == B(): pass # Noncompliant {{Remove this equality check between incompatible types; it will always return False.}}
    if A() != B(): pass # Noncompliant {{Remove this equality check between incompatible types; it will always return True.}}
    if A() >= B(): pass
    if A() == ClassWithEq(): pass
    if ClassWithEq() == A(): pass
    if A() != ClassWithNe(): pass
    if ClassWithNe() != A(): pass


    if A() == 42: pass # Noncompliant
    if 'a' == 42: pass # Noncompliant
    if 'a' == 42: pass # Noncompliant
    if A() == []: pass # Noncompliant
    if A() == (): pass # Noncompliant
    if A() == {}: pass # Noncompliant
    if {42:''} == {1}: pass # Noncompliant
    if A() == None: pass # handled by S5727
    if None == A(): pass # handled by S5727
    if 'a' == True: pass # Noncompliant
    if 'a' == 42.42: pass # Noncompliant
    if 'a' == 42j: pass # Noncompliant
    if 'foo'.partition('f') == "42": pass # Noncompliant
    if 1 == True: pass
    if ClassWithEq() == 42: pass
    if ClassWithEq() != 42: pass
    if ClassWithNe() != 42: pass
    if 42 == ClassWithEq(): pass
    if 42 != ClassWithEq(): pass
    if 42 != ClassWithNe(): pass
    if set([1, 2]) == frozenset([1, 2]): pass
    if {} == collections.OrderedDict(): pass

def stdlib():
    if acos(1) == "0": pass # Noncompliant
    if math.ceil(7.9) == "8": pass # FN: ceil defined twice in math for python 2 & python 3
    if math.pow(2,3) == "8": pass # FN pow defined in builtin as well as math
    if iskeyword("something") == "1": pass # Noncompliant
    tuple = "name", "passwd", 123, 456, "gecos", "dir", "shell"
    passwd = pwd.struct_passwd(tuple)
    if passwd == "something": pass
    passwd_2 = pwd.struct_passwd(tuple)
    if passwd == passwd_2: pass # OK
    if passwd == pwd.getpwuid(1): pass # OK
    if 42 == pwd.getpwuid(1): pass # Noncompliant
    if pwd.getpwall() == 42: pass # Noncompliant
    if zip(l1, l2) == 42: pass # FN due to missing Python 2 and usage of zip.__new__
    if platform.architecture() == '32bit': ... # Noncompliant

def third_party():
  if emojize("Python is :thumbs_up:") == 42: ... # Noncompliant
