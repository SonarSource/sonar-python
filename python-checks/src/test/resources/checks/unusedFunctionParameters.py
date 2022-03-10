def f(unread_param):  # Noncompliant {{Remove the unused function parameter "unread_param".}}
#     ^^^^^^^^^^^^
    print("foo")


def g(read_param):
    print(read_param)


def f(unread_param):
    return locals()


def g(read_param):  # Noncompliant
    a = 1
    b = 2
    compute(a)

class MyInterface:

    def write_alignment(self, alignment):
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
