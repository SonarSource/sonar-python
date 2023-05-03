assert x < 2
assert (x < 2) #Noncompliant {{Remove the parentheses after this "assert" keyword.}}

del x
del(x) #Noncompliant {{Remove the parentheses after this "del" keyword.}}

if x < 2:
	pass
elif x > 3:
	pass

if (x < 2): #Noncompliant {{Remove the parentheses after this "if" keyword.}}
	pass
elif (x > 3): #Noncompliant {{Remove the parentheses after this "elif" keyword.}}
	pass

if (x > 2) and (x < 10): # Compliant with binary operators
	pass
elif (x >= 10) and (x < 20): # Compliant with binary operators
	pass

for x in range(0, 3):
	pass

for (x) in range(0, 3): #Noncompliant {{Remove the parentheses after this "for" keyword.}}
	pass

for x in (range(0, 3)): #Noncompliant {{Remove the parentheses after this "in" keyword.}}
	pass

if x:
	raise NameError('HiThere')

if x:	
	raise (NameError('HiThere')) #Noncompliant {{Remove the parentheses after this "raise" keyword.}}

if x:
	raise

def func():
	return x

def func():
	return (x) #Noncompliant {{Remove the parentheses after this "return" keyword.}}

def func(x):
    return (x.a, x.b) #Noncompliant
def func(x):
    return (x.a,
            x.b)
def func(x):
    return x.a, x.b


def func():
	return [1, 2]

def func():
	return

def func(x):
  return () # compliant, empty tuple
def func(x):
	return 1, 2

def func():
    return (1,) #compliant, single element in tuple

def func(x):
	return '' if x is None else x.isoformat()

while x < 2:
	pass

while (x < 2): #Noncompliant {{Remove the parentheses after this "while" keyword.}}
	pass
yield
yield x
yield (x) #Noncompliant {{Remove the parentheses after this "yield" keyword.}}
yield (a, b) #Noncompliant
yield a, (b,c) #compliant
yield (a,b), c #compliant
yield (a,) #compliant

try:
	x = 1
except ValueError:
	pass

try:
	x = 1
except (ValueError): #Noncompliant {{Remove the parentheses after this "except" keyword.}}
	pass

try:
	x = 1
except (ValueError, TypeError):
	pass

if not x:
	pass

if not(x): #Noncompliant {{Remove the parentheses after this "not" keyword.}}
	pass

if not(x and y):
	pass

if not(x <= 200 < y): # Noncompliant
  pass

for j in (s.id.strip() for s in alignment):
    print j
# Line continuation: parenthesis are required to be able to split the line
if (x > 0 and 
	x < 3):
	pass

my_pairs = [(1, 2), (5, 6)]
names = ['small', 'large']
for (first, second), name in zip(my_pairs, names): # the parenthesis after the "for" keyword is not useless
    print(name, first + second)

for (x, y) in foo: # Noncompliant {{Remove the parentheses after this "for" keyword.}}
    print(x, y)

for (x, ) in foo: # compliant
    print(x)

for x, y in (foo): # Noncompliant {{Remove the parentheses after this "in" keyword.}}
    pass

for x in ("who", ): # compliant
	pass
for x in ("who", "comments", "revlink", "category", "branch", "revision"): # Noncompliant
	pass
for x in ("who", "comments", "revlink", "category", "branch", "revision"), foo:
	pass

a = (10, 5)
b = (1,)

# SONARPY-292 should not raise issues on tuples
my_tuple_list = [('foo', 'bar')]
if ('foo', 'bar') in my_tuple_list:
	print("True")

# SONARPY-1021 should not raise issues on assignment expression (walrus operator)
if (x := 3):
    ...
if not (x := 3):
    ...
x = [(y := 3) for num in numbers if value > 0]
yield (x := 3)
assert (x := 3)
while (x := 3):
    pass

def fooWalrus(val):
    return (x := val)

