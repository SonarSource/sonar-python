for i in range(1):
    break

for i in range(1):
    break; print(i) # Noncompliant {{Remove the code after this "break".}}
#   ^^^^^

for i in range(1):
    continue # Noncompliant {{Remove the code after this "continue".}}
    print(i)

if True:
    print(1)
    raise TypeError("message") # Noncompliant {{Remove the code after this "raise".}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^
    if True: pass

def fun1():
    return 1

def fun2():
    print(1)
    return 2

def fun3():
    return 2 # Noncompliant {{Remove the code after this "return".}}
    print(1)

def fun4():
    return 2;

def fun5():
    return 2; print(1) # Noncompliant {{Remove the code after this "return".}}
