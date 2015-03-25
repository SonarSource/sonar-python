x = (1 + 1)

x = (2 + 2) * 1

def fun():
    return(x)

print("Hello")

assert (x < 2)

del(x)

if x < 2:
    pass
elif x > 3:
    pass

if (x < 2):
    pass
elif (x > 3):
    pass

for x in range(0, 3):
    pass

for (x) in range(0, 3):
    pass

for x in (range(0, 3)):
    pass

if x:
    raise NameError('HiThere')

if x:
    raise (NameError('HiThere'))

while (x < 2):
    pass

try:
    x = 1
except (ValueError):
    pass

try:
    x = 1
except (ValueError, TypeError):
    pass

if not x:
    pass

if not(x):
    pass

if not(x and y):
    pass

# Line continuation: parenthesis are required to be able to split the line
if (x > 0 and
            x < 3):
    pass

for (i, j) in enumerate(d):
    pass

return (a, b)

x = (y,)

yield (1,2)

return "", (1,)

return name, (value, params)

for x, (y, z) in range(0, 3):
    pass
