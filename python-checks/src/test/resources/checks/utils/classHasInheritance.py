
class A(object): # Noncompliant {{no_inheritance}}
    field = 1

class B(): # Noncompliant {{no_inheritance}}
    field = 1

class C(A): # Noncompliant {{has_inheritance}}
    field = 1

class D(object, A): # Noncompliant {{has_inheritance}}
    field = 1

class E: # Noncompliant {{no_inheritance}}
    field = 1
