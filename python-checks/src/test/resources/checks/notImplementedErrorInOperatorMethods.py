class AllMethods(object):
    def __lt__(self, other):
        raise NotImplementedError() # Noncompliant {{Return "NotImplemented" instead of raising "NotImplementedError"}}
       #^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def __le__(self, other):
        raise NotImplementedError() # Noncompliant
    def __eq__(self, other):
        raise NotImplementedError() # Noncompliant
    def __ne__(self, other):
        raise NotImplementedError() # Noncompliant

    def __gt__(self, other):
        raise NotImplementedError # Noncompliant
    def __add__(self, other):
        raise ValueError
    def __sub__(self, other):
        raise ThereIsNoSuchException

    def __mul__(self, other):
        ex = NotImplementedError()
        raise ex # Noncompliant

    def foo(self):
        raise NotImplementedError()

    def add(self, other):
        raise NotImplementedError()

    def bar(self):
        pass

class CompliantClass(object):
    def __lt__(self, other):
        return NotImplemented
    def __le__(self, other):
        return NotImplemented
    def __eq__(self, other):
        return NotImplemented
    def __ne__(self, other):
        return NotImplemented

class NumberReturns(object):
    def __add__(self, other):
        return 42
    def __radd__(self, other):
        return 42

class BareRaises(object):
    def __add__(self, other):
        raise

class OtherRaises(object):
    def __add__(self, other):
        raise ValueError()

def foo(a, b):
    raise NotImplementedError()

def __lt__(a, b):
    raise NotImplementedError()

class WithParameters(object):
    def __add__(self, other):
        raise NotImplementedError("This is just to see if we trigger with constructor arguments") # Noncompliant
       #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
