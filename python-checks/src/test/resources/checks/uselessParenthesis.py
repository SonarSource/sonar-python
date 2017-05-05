x = 1 + 2

x = (1 + 2)

x = ((1 + 2))                          # Noncompliant [[secondary=+0]] {{Remove those useless parentheses.}}
#   ^

x = ((((1 + 2))))
# Noncompliant@-1 [[sc=5;ec=6;secondary=+0]]
# Noncompliant@-2 [[sc=6;ec=7;secondary=+0]]
# Noncompliant@-3 [[sc=7;ec=8;secondary=+0]]

y = ((x1) + (x2))

y = ((x1))
# Noncompliant@-1
 
y = (x1) + (x2)
 
y = ((x1)) + ((x2))
# Noncompliant@-1
# Noncompliant@-2

y = x1 * ((x3))
# Noncompliant@-1

y = (x1 * (x2 + 2) * ((x3 + 3)) + 4)
# Noncompliant@-1 [[sc=22]]

# tuple:
y = ((x1, x2))                         # Noncompliant
y = ((x,))                             # Noncompliant
y = ((x), )

# function with a tuple argument: double parentheses are required
n = len((x, y))
n = len((x,))

# double parentheses may be required, depending on the signature of foo (tuple vs list of arguments)
a = foo((x, y))
a = foo(((x, y)))                      # Noncompliant
a.append((x, y))
a.append(((x, y)))                     # Noncompliant

y = foo({x, y})
y = foo(({x, y}))
y = foo((({x, y})))                    # Noncompliant

def fun1():
    return(x)

def fun2():
    return((x))                        # Noncompliant

print("Hello")
print(("Hello"))
print((("Hello")))                     # Noncompliant

assert (x < 2)
assert ((x < 2))                       # Noncompliant

del((x))                               # Noncompliant

query = query.where((x != y))          # FN

if (x < 2):
    pass
elif (x > 3):
    pass

if ((x < 2)):                          # Noncompliant
    pass
elif ((x > 3)):                        # Noncompliant
    pass

if isinstance(x, (basestring)):
    pass

for (x) in (range(0, 3)):
    pass

for ((x)) in (range(0, 3)):            # Noncompliant
#   ^
    pass

for (x) in ((range(0, 3))):            # Noncompliant
#          ^
    pass

for candidate in (((d not in models) for d in deps)): #Noncompliant
    pass

for candidate in (((d not in models), d,)):           #Noncompliant
    pass

for candidate in (((d not in models) ,*d,)):          #Noncompliant
    pass

if x:
    raise (NameError('Hithere'))

if x:
    raise ((NameError('Hithere')))     # Noncompliant
#         ^

while ((x < 2)):                       # Noncompliant
    pass

platform = ((sys.platform in ('win32', 'Pocket PC'))) # Noncompliant

try:
    x = 1
except ((ValueError)):                 # Noncompliant
    pass

try:
    x = 1
except ((ValueError, TypeError)):      # Noncompliant
    pass

if not(x):
    pass

if not((x)):                           # Noncompliant
    pass

if not((x and y)):                     # Noncompliant
    pass

# Line continuation: parenthesis are required to be able to split the line
if ((x > 0 and                         # Noncompliant 
            x < 3)):
    pass

for ((i, j)) in enumerate(d):          # Noncompliant
    pass

return ((a, b))                        # Noncompliant

yield ((1,2))                          # Noncompliant

gen = (x*x for x in range(10))
gen = ((x*x for x in range(10)))       # Noncompliant

return "", ((1,))                      # Noncompliant

return name, ((value, params))         # Noncompliant

for x, ((y, z)) in range(0, 3):        # Noncompliant
    pass

req_path = (parsed.path or '/') + (('?' + parsed.query) if parsed.query else '')

encode = lambda k, v: '%s=%s' % ((quote(k, safe), quote(v, safe)))   # Noncompliant
