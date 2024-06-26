def fun1():
    if True:
        return 1
    else:
        def fun2(): # Noncompliant {{This function has 3 returns or yields, which is more than the 2 allowed.}}
#           ^^^^
            if True:
                return 1
#               ^^^^^^^^< {{"return" statement.}}
            elif False:
                return 2
#               ^^^^^^^^< {{"return" statement.}}
            else:
                return 3
#               ^^^^^^^^< {{"return" statement.}}
        return 2


def fun3(): # Noncompliant 
#   ^^^^
    for i in range(5):
        if i > 0:
            return 1
#           ^^^^^^^^<
    if True:
        return 2
#       ^^^^^^^^<
    return 3
#   ^^^^^^^^<


def fun4(): # Noncompliant
    for i in range(5):
        yield i
    yield 4
    yield 5
    return
