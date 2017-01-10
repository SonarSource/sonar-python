expression = 3

class MyClass: # Noncompliant [[effortToFix=2]] {{Class has a complexity of 4 which is greater than 2 authorized.}}
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
        if expression:
            pass
        return
