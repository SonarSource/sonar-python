DEBUG = True  # Noncompliant
DEBUG_PROPAGATE_EXCEPTIONS = True  # Noncompliant
DEBUG += True  # OK
DEBUG = False  # OK
DEBUG = print()  # OK
DEBUG_PROPAGATE_EXCEPTIONS = False  # OK
Other = True

class MyClass:
    def __init__(self):
        self.boolean_property = True
my_object = MyClass()
my_object.boolean_property = True
