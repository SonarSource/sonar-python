class A:
    @property
    def foo(self, unexpected): # Noncompliant {{Remove 1 parameters; property getter methods receive only "self".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^
        return self._foo

    @foo.setter
    def foo(self, value, unexpected): # Noncompliant {{Remove 1 parameters; property setter methods receive "self" and a value.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self._foo = value

    @foo.deleter
    def foo(self, unexpected): # Noncompliant {{Remove 1 parameters; property deleter methods receive only "self".}}
        del self._foo

class B:
    def get_foo(self, unexpected):  # Noncompliant {{Remove 1 parameters; property getter methods receive only "self".}}
        return self._foo

    def set_foo(self):  # Noncompliant {{Add the value parameter; property setter methods receive "self" and a value.}}
        self._foo = 1

    def del_foo(self, unexpected):  # Noncompliant {{Remove 1 parameters; property deleter methods receive only "self".}}
        del self._foo

    foo = property(get_foo, set_foo, del_foo, "'foo' property.")

class C:
    @property
    def foo(self):
        return self._foo

    @foo.setter
    def foo(self, value):
        self._foo = value

    @foo.deleter
    def foo(self, glob=1): # OK, default arguments do not count
        del self._foo

class D:
    def get_foo(self):
        return self._foo

    def set_foo(self, value=None): # Ok
        self._foo = value

    def del_foo(self):
        del self._foo

    def del_foo2(self, *, k1, k2): # Noncompliant
        pass

    foo = property(get_foo, set_foo, del_foo, "'foo' property.")
    foo2 = property(get_foo, set_foo, del_foo2)

    def get_bar(self):
        return self._bar

    def set_bar(self, value, unexpected): # FN
        self._bar = value

    bar = property(fset=set_bar, fget=get_bar)

class EdgeCase:
    def get_foo(self):
        pass

    def set_foo(self, value):
        pass

    def del_foo(self):
        pass
    del_foo = 1

    def set_foo2(self, (x, y)):
        pass

    def get_foo2():
        pass

    foo = property(get_foo, there_is_no_set_foo, 1 + 1)
    foo2 = property(*(get_foo, set_foo))
    foo3 = property(get_foo, set_foo, del_foo)
    foo4 = unknown_callee(get_foo, set_foo, del_foo)
    foo5 = property(get_foo2, set_foo2)

    def some_function(x, y):
        return x
    some_function = 2

    foo6 = some_function(get_foo2, set_foo2)

    @staticmethod
    def static_method():
        pass

    @some.decorator
    def another_fun(self):
        pass

    @bar.setter
    def bar(self, value):
        self._foo = value

    @property
    def bar(self):
        return self._foo

    p = property({}.__getitem__)
