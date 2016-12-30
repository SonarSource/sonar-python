expression = 3

class MyClass: # Noncompliant [[effortToFix=3]] {{Class has a complexity of 5 which is greater than 2 authorized.}}
#     ^^^^^^^
    def hello(self):
        if expression:
            pass
        if expression:
            pass
        if expression:
            pass
        return

class MyClass2:
    def hello(self):
        return
