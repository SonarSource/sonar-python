if 3 > 2:
    print "x"
else:
    print "xx"

# Noncompliant@+3 {{Either remove or fill this block of code.}} 
# Noncompliant@+5
if 3 > 2:
    pass
#   ^^^^
else:
    pass
#   ^^^^

if 3 > 2:
    print "x"
    pass

if 3 > 2:
    if 3 > 2:
        print "x"

if 3 > 2: print "x"
# Noncompliant@+1
if 3 > 2: pass

# Noncompliant@+2
try:
    pass
except:
    pass

if 3 > 2:
    # nothing to do
    pass

if 3 > 2:
    pass  # nothing to do

if 3 > 2:
   # no issue
   pass
elif k == 5: # unknown
    pass

def empty_function():
    pass


class empty_class:
    pass
# Noncompliant@+2
if condition1:
    pass
# just a comment
elif condition2:
    foo()
