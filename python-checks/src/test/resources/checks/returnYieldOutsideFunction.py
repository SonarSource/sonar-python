class MyClass:
    while True:
        return False # Noncompliant {{Remove this use of "return".}}
#       ^^^^^^^^^^^^

    def func(self):
        while True:
            yield 1


def func():
    class A:
        return True # Noncompliant
    return 1


yield True # Noncompliant {{Remove this use of "yield".}}
