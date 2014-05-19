assert x < 2
assert (x < 2)

del x
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

if x:
	raise

def func():
	return x

def func():
	return (x)
	
while x < 2:
	pass

while (x < 2):
	pass

yield x
yield (x)
	
try:
	x = 1
except ValueError:
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
