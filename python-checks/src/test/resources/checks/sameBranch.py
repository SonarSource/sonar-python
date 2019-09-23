a = 1
param = 1

if 0 <= a < 10:
    print(1)
    foo()
elif 10 <= a < 20:
    print(2)
elif 20 <= a < 50:
    print(1) # Noncompliant [[sc=5;ec=10;el=+1;secondary=-5]]
    foo()
else:
    print(3)

if param == 1:
    print(1)
    foo()
else:
    if param == 2:
        print(1) # Noncompliant [[secondary=-4]]
        foo()

if param == 1:
    if True:
        pass
else:
    if param == 2:
        print(2)
    else:
        if True: # Noncompliant [[secondary=-6]]
            pass

if True:
    if True:
        print(1)
    else:
        print(1)  # Noncompliant {{Either merge this branch with the identical one on line "35" or change one of the implementations.}}

if param == 1:
    print(1)
else:
    if param == 2:
        print(1)
    print(2)

if 1: print("1"); foo()
elif 2: print("1"); foo()
else: print("2")

if 1:
    print("1")
elif 2:
    print("2")
else:
    print("1")

if 1:
    print("1")
elif 2:
    print("1")
else:
    print("1") # Noncompliant

if 1:
    print("1")
    foo()
elif 2:
    print("1") # Noncompliant [[secondary=-3;sc=5;ec=10;el=+1]]
    foo()
else:
    print("2")

a = 1 if 1 else 2
a = 1 if x else 1 # Noncompliant [[secondary=+0]]
#               ^
a = 1 if x else 2 if y else 2
a = 2 if x else 1 if y else 2
a = 2 if x else 2 if y else 2 # Noncompliant
#                           ^
a = 2 \
    if x else 2 if y \
    else 2
# Noncompliant@-1
a = 2 if x else (t + 4) if y else (t +
                               4)
# Noncompliant@-2
a = (1 if x else 3) if y else 5

a = 1 if x else (1) # Noncompliant
#                ^
a = ((1)) if x else (1) # Noncompliant
#                    ^
a = ((1)) if x else 1 # Noncompliant
#                   ^
a = 2 if x else ((2) if y else 2) # Noncompliant
#                              ^
a = (1, 2) if x else (1, 3)
a = (1, 2) if x else (1, 2) # Noncompliant
#                    ^^^^^^
a = [1] if x else [2]
a = [1] if x else [1] # Noncompliant
#                 ^^^

if x in ('a', 'b'):
    print("1")
    print("2")
elif x == 'c':
    print("1") # Noncompliant
    print("2")

def fun(self, other):
    if self.changes and other.changes:
        return True
    elif self.changes and not other.changes:
        return False
    elif not self.changes and other.changes:
        return False
