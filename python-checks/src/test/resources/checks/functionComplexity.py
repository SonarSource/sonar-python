expression = 3

def hello(): # Noncompliant [[effortToFix=3]] {{Function has a complexity of 5 which is greater than 2 authorized.}}
#   ^^^^^
    if expression:
        pass
    if expression:
        pass
    if expression:
        pass
    return

def hello2():
    return
