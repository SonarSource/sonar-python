class MyClass: # Noncompliant {{Class has a complexity of 5 which is greater than 2 authorized.}}
#     ^^^^^^^
    def hello():
        if expression:
            pass
        if expression:
            pass
        if expression:
            pass
        return

class MyClass2:
    def hello():
        return
