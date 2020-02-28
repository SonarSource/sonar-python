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
    if A() == None: pass # Noncompliant
    if 'a' == True: pass # Noncompliant
    if 'a' == 42.42: pass # Noncompliant
    if 'a' == 42j: pass # Noncompliant
    if 1 == True: pass
    if ClassWithEq() == 42: pass
    if ClassWithEq() != 42: pass
    if ClassWithNe() != 42: pass
    if 42 == ClassWithEq(): pass
    if 42 != ClassWithEq(): pass
    if 42 != ClassWithNe(): pass
    if set([1, 2]) == frozenset([1, 2]): pass
