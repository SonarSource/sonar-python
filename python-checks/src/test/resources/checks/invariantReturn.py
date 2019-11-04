def f01():
    return 1

def f02(x): # Noncompliant {{Refactor this method to not always return the same value.}}
#   ^^^
    if x:
        return None
#       ^^^^^^^^^^^<
    else:
        return None
#       ^^^^^^^^^^^<

def f03(x): # Noncompliant
#   ^^^
    if x:
        return -(+5 * 2)
#       ^^^^^^^^^^^^^^^^<
    else:
        return -(+5 * 2)
#       ^^^^^^^^^^^^^^^^<

def f04(x):
    if x:
        return 1
    else:
        return 2

def f05(x):
    if x:
        return 1
    return 2

def f06(x):
    pass

def f07(x):
    if x:
        return 1
    foo()

def f08(x): # Noncompliant
#   ^^^
    if x:
        return ""
#       ^^^^^^^^^<
    else:
        return ""
#       ^^^^^^^^^<

def f09(x):
    if x:
        return "" + ""
    else:
        return "" + "b"

def f09(x, a): # false-negative
    if x:
        return "" + a
    else:
        return "" + a

def f10(): # Noncompliant
    try:
        f09(2)
    except Error as e:
        return 1
    return 1

def f11():
    try:
        return 0
    except:
        return 0
    return 1

def f12():
    try:
        return 1
    except:
        return 0
    except:
        return 0
    return 0

def f13(): # Noncompliant
    try:
        return 0
    except A:
        foo()
    finally:
        x = y
    return 0

def f14(x): # Noncompliant
#   ^^^
    if x:
        try:
            return "ignored because of finally"
        except:
            return "ignored because of finally"
        finally:
            return 0
#           ^^^^^^^^<
    else:
        return 0
#       ^^^^^^^^<

def f15(x):
    if x:
        try:
            return 0
        except Error as e:
            return 1
        except:
            return 0
    else:
        return 0


def f16(x): # Noncompliant
#   ^^^
    for i in x:
        return 0
#       ^^^^^^^^<
    return 0
#   ^^^^^^^^<

def f17(x): # Noncompliant
#   ^^^
    for i in x:
        return 0
#       ^^^^^^^^<
    for i in x:
        if i == 2:
            break
        else:
            return 0
#           ^^^^^^^^<
    else:
        return 0
#       ^^^^^^^^<
    if x:
        try:
            return 0
#           ^^^^^^^^<
        except Error as e:
            return 0
#           ^^^^^^^^<
        except:
            return 0
#           ^^^^^^^^<
    elif x == 2:
        return 0
#       ^^^^^^^^<
    else:
        return 0
#       ^^^^^^^^<

def f18(x):
    if x:
        return
    return

def f19(x):
    if x:
        return True
    return

def f20(x):
    if x:
        return 1, 2
    return 1

def f21(x):
    if x:
        return 1 + a
    return a + 1
