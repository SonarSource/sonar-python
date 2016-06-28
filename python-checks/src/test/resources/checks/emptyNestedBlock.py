if 3 > 2:
    print "x"
else:
    print "xx"

# Noncompliant@+3 [[sc=5;ec=9]] {{Either remove or fill this block of code.}}
# Noncompliant@+4
if 3 > 2:
    pass
else:
    pass

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


def empty_function():
    pass


class empty_class:
    pass
