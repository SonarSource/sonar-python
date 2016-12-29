a = 1
b = 1

def work():
    pass

if a == a: # Noncompliant [[secondary=+0;sc=9;ec=10]] {{Identical sub-expressions on both sides of operator "==".}}
#       ^
    work()

if  a != \
        a: # Noncompliant [[secondary=-1;sc=9;ec=10]]
    work()

if  a == b and a == b: # Noncompliant
    work()

if a == b or a == b: # Noncompliant
#            ^^^^^^
    work()

j = 5 / 5 # Noncompliant
k = 5 - 5 # Noncompliant
exclusion = 1 << 1
exclusion2 = (a * b) << 1
