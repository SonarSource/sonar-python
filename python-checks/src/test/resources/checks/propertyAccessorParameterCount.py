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
    def foo(self):
        del self._foo

class D:
    def get_foo(self):
        return self._foo

    def set_foo(self, value):
        self._foo = value

    def del_foo(self):
        del self._foo

    foo = property(get_foo, set_foo, del_foo, "'foo' property.")

class EdgeCase:
    def get_foo(self):
        pass

    def set_foo(self, value):
        pass

    def del_foo(self):
        pass
    del_foo = 1

    foo = property(get_foo, there_is_no_set_foo, 1 + 1)
    foo2 = property(*(get_foo, set_foo))
    foo3 = property(get_foo, set_foo, del_foo)
    foo4 = unknown_callee(get_foo, set_foo, del_foo)

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
