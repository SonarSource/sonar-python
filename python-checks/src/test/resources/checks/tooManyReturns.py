def fun1():
    if True:
        return 1
    else:
        def fun2():
            if True:
                return 1
            elif False:
                return 2
            else:
                return 3
        return 2


def fun3():
    for i in range(5):
        if i > 0:
            return 1
    if True:
        return 2
    return 3
