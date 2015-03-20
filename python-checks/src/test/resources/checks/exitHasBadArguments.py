class MyClass1:
    def __enter__(self):
        pass
    def __exit__(self, (exc_type, exc_val)):
        pass

class MyClass2:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_trace, exc_one_more):  # Noncompliant
        pass

class MyClass3:
    def __enter__(self):
        pass
    def __exit__(self):
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
    def __exit__():
        pass
