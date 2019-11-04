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
   #^[el=+2;ec=12]>
        pass
else:
    if param == 2:
        print(2)
    else:
        if True: # Noncompliant
       #^[el=+2;ec=16]
            pass

if cond:
  foo()
  bar()
elif cond:
  foo() #Noncompliant
  bar()
elif cond:
  foo() #Noncompliant
  bar()

if True:
#In this case, S3923 will raise a bug
    if True:
        print(1)
    else:
        print(1)

if param == 1:
    print(1)
else:
    if param == 2:
        print(1)
    print(2)

if 1: print("1"); foo()
elif 2: print("1"); foo()
else: print("2")

#ok, no issue raised for branches with 1 line of code
if 1:
    print("1")
elif 2:
    print("2")
else:
    print("1")

#exception: all branches identical without else clause
if cond:
  foo()
elif cond:
  foo() # Noncompliant
elif cond:
  foo() # Noncompliant

#In this case, S3923 will raise a bug
if 1:
    print("1")
elif 2:
    print("1")
else:
    print("1")

if 1:
    print("1")
    foo()
elif 2:
    print("1") # Noncompliant [[secondary=-3;sc=5;ec=10;el=+1]]
    foo()
else:
    print("2")

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

def fun():
    for element in iterable:
        if (hasattr(element, '__iter__') and
                not isinstance(element, basestring)):
            for f in _izip_fields(element):
                yield f
        elif isinstance(element, np.void) and len(tuple(element)) == 1:
            for f in _izip_fields(element): # Noncompliant
                yield f
        else:
            yield element
