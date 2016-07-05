if True:
    pass
else:
    print(1)
    pass # Noncompliant {{Remove this unneeded "pass".}}
#   ^^^^

def fun1():
    if True:
        print(1)
    pass  # Noncompliant

def fun2():
    pass

def fun3():
    print(1); pass # Noncompliant
#             ^^^^

def fun4():
    pass  # Noncompliant
    print(1)

def fun5(): pass

def fun6(): print(1); pass  # Noncompliant

def fun7():
    '''
    docstring is the only statement that could appear with pass
    '''
    pass

pass
