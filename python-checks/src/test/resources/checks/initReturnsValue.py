class MyClass1:
    def __init__(self):
        return None

class MyClass2:
    def __init__(self):
        yield None # Noncompliant {{Remove this yield statement.}}
#       ^^^^^^^^^^

class MyClass3:
    def __init__(self):
        yield 1   # Noncompliant

class MyClass4:
    def __init__(self):
        return

class MyClass5:
    def __init__(self):
        return fun()   # Noncompliant {{Remove this return value.}}
#       ^^^^^^^^^^^^

class MyClass6:
    def __init__(self):
        def fun():
            return 1
        self.count = fun()

