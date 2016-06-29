
def fun1(n): # Noncompliant {{Use only "return" or only "yield", not both.}}
#   ^^^^
    num = 0
    while num < n:
        yield num
        num += 1
    return num


def fun2(n):
    num = 0
    while num < n:
        yield num
        num += 1
    return


def fun3(n):
    num = 0
    while num < n:
        yield num
        num += 1

def fun5(n): # Noncompliant
    num = 0
    if n == 0:
        return
    while num < n:
        yield num
        num += 1
    return num

def fun4(n):
    num = 0
    while num < n:
        num += 1
    return num

def fun6():
    def fun7():
        yield 1
    return 1

def fun8():
    def fun9():
        return 1
    yield 1
