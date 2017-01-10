def hello1():      # Noncompliant {{Function has a complexity of 16 which is greater than 15 authorized.}}
    if (1 == 2 and 1 == 2 and 1 == 2 and 1 == 2 and 1 == 2):
        pass
    if (1 == 2 and 1 == 2 and 1 == 2 and 1 == 2 and 1 == 2):
        pass
    if (1 == 2 and 1 == 2 and 1 == 2 and 1 == 2 and 1 == 2):
        pass
    return

def hello2():      # OK, complexity is 11
    if (1 == 2 and 1 == 2 and 1 == 2 and 1 == 2 and 1 == 2):
        pass
    if (1 == 2 and 1 == 2 and 1 == 2 and 1 == 2 and 1 == 2):
        pass
    return
