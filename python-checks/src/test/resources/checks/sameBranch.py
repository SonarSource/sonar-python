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
