def f(a):  # Noncompliant {{Remove the unused function parameter "a".}}
#     ^
    print("foo")


def g(a):
    print(a)


def f(a):
    return locals()


def g(a):  # Noncompliant
    b = 1
    c = 2
    compute(b)

class MyInterface:

    def write_alignment(self, a):
        """This method should be replaced by any derived class to do something
        useful.
        """
        raise NotImplementedError("This object should be subclassed")

class Parent:
    def do_something(self, a, b):  # Noncompliant {{Remove the unused function parameter "a".}}
        #                  ^
        return compute(b)

    def do_something_else(self, a, b):
        return compute(a + b)


class Child(Parent):
    def do_something_else(self, a, b):
        return compute(a)


class AnotherChild(UnknownParent):
    def _private_method(self, a, b):  # Noncompliant
        return compute(b)

    def is_overriding_a_parent_method(self, a, b):
        return compute(b)

class ClassWithoutArgs:
    def do_something(self, a, b):  # Noncompliant
        return compute(b)

class ClassWithoutArgs2():
    def do_something(self, a, b):  # Noncompliant
        return compute(b)
