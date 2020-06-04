def process(object=[]):  # Noncompliant {{Rename this parameter; it shadows a builtin.}}
#           ^^^^^^
    pass

def a(max, /):  # Noncompliant {{Rename this parameter; it shadows a builtin.}}
    # ^^^
    pass

lambda max: max # Noncompliant {{Rename this parameter; it shadows a builtin.}}
#      ^^^

class MyClass:
    def method(self, max=1):  # Noncompliant {{Rename this parameter; it shadows a builtin.}}
        #            ^^^
        pass

class SubClass(MyClass):
    def method(self, max=1):  # Ok even if max is a builtin. Liskov Substitution Principle is more important.
        pass
