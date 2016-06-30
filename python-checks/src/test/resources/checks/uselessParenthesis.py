x = (1 + 1) # Noncompliant {{Remove those useless parentheses}}
#   ^^^^^^^
x = (2 + 2) * 1

def fun():
    return(x) # Noncompliant

print("Hello")

assert (x < 2) # Noncompliant

del(x) # Noncompliant

if x < 2:
    pass
elif x > 3:
    pass

if (x < 2): # Noncompliant
    pass
elif (x > 3): # Noncompliant
    pass

for x in range(0, 3):
    pass

for (x) in range(0, 3): # Noncompliant
    pass

for x in (range(0, 3)): # Noncompliant
#        ^^^^^^^^^^^^^
    pass

if x:
    raise NameError('HiThere')

if x:
    raise (NameError('HiThere')) # Noncompliant
#         ^^^^^^^^^^^^^^^^^^^^^^

while (x < 2): # Noncompliant
    pass

try:
    x = 1
except (ValueError): # Noncompliant
    pass

try:
    x = 1
except (ValueError, TypeError):
    pass

if not x:
    pass

if not(x): # Noncompliant
    pass

if not(x and y):
    pass

# Line continuation: parenthesis are required to be able to split the line
if (x > 0 and
            x < 3):
    pass

for (i, j) in enumerate(d): # Noncompliant
    pass

return (a, b) # Noncompliant

x = (y,)

yield (1,2) # Noncompliant

return "", (1,)

return name, (value, params)

for x, (y, z) in range(0, 3):
    pass
