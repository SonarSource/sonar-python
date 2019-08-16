class MyClass1:
    def __enter__(self):
        pass
    def __exit__(self, (exc_type, exc_val)): # Noncompliant {{Add the missing argument.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        pass

class MyClass2:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_trace, exc_one_more):  # Noncompliant {{Remove the unnecessary argument.}}
        pass

class MyClass3:
    def __enter__(self):
        pass
    def __exit__(self):  # Noncompliant
        pass

class MyClass4:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MyClass7:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

class MyClass5:
    def __enter__(self):
        pass
    def __exit__(self, *args, **args2):
        pass

class MyClass6:
    def __enter__(self):
        pass
    def __exit__():  # Noncompliant
        pass
