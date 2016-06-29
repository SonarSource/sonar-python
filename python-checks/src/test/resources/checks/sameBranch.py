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
else:
    if param == 2:
        print(1) # Noncompliant [[secondary=-3]]


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
        print(1)  # Noncompliant {{Either merge this branch with the identical one on line "34" or change one of the implementations.}}

if 1: print("1"); foo()
# Noncompliant@+1 [[secondary=-2;sc=9;ec=26;el=+0]]
elif 2: print("1"); foo()
else: print("2")
