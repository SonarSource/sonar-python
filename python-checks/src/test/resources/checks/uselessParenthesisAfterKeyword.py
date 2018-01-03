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
	
while x < 2:
	pass

while (x < 2): #Noncompliant {{Remove the parentheses after this "while" keyword.}}
	pass

yield x
yield (x) #Noncompliant {{Remove the parentheses after this "yield" keyword.}}
	
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

# Line continuation: parenthesis are required to be able to split the line
if (x > 0 and 
	x < 3):
	pass

my_pairs = [(1, 2), (5, 6)]
names = ['small', 'large']
for (first, second), name in zip(my_pairs, names): # the parenthesis after the "for" keyword is not useless
    print(name, first + second)

for (x, y) in foo: # Noncompliant
    print(x, y)
