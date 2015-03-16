a = 1
param = 1

if 0 <= a < 10:
    print(1)
elif 10 <= a < 20:
    print(2)
elif 20 <= a < 50:
    print(1)
else:
    print(3)

if param == 1:
    print(1)
else:
    if param == 2:
        print(1)


if param == 1:
    print(1)
else:
    if param == 2:
        print(2)
    else:
        print(1)

if True:
    if True:
        print(1)
    else:
        print(1)
