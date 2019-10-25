expression = 3

def hello(): # Noncompliant {{Function has a complexity of 4 which is greater than 2 authorized.}} [[effortToFix=2]]
#   ^^^^^
    if expression:
        pass
    if expression:
        pass
    if expression:
        pass
    return

def hello2():
    if expression:
        pass
    return
