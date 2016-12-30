a = 1
b = 1

def work():
    pass

if a == a: # Noncompliant [[secondary=+0;sc=9;ec=10]] {{Correct one of the identical sub-expressions on both sides of operator "==".}}
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
l = 5 + 5
m = 5 * 5
n = 3 << 3 # Noncompliant
o = 3 & 3 # Noncompliant
p = 3 ^ 3 # Noncompliant
q = 3 | 3 # Noncompliant
r = 3 and 3 # Noncompliant
s = 3 or 3 # Noncompliant
c1 = 3 < 3 # Noncompliant
c2 = 3 <= 3 # Noncompliant
c3 = 3 > 3 # Noncompliant
c4 = 3 >= 3 # Noncompliant
c5 = 3 <> 3 # Noncompliant
c6 = 3 is 3 # Noncompliant
c7 = 3 is not 3 # Noncompliant
c8 = 3 not in 3 # Noncompliant
c9 = 3 in 3 # Noncompliant
exclusion = 1 << 1
exclusion2 = (a * b) << 1
