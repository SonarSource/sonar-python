#!/usr/bin/python

# Python exception handling.

# Choose two random integers.
import random
i = random.randrange(0, 8)
j = random.randrange(-1, 6)
print i, j

# Get a nice little array, then try a bunch of dangerous stuff.
some = [3, 10, 0, 8, 18];
try:
    # We try to execute this block.
    den = some[j] / i
    print "A:", den
    frac = (i + j) / den
    print "B:", frac
    if frac < 2:
        k = 3
    else:
        k = 'mike'
    print "C:", k
    print "D:", some[k]
# This is the catch block.
except ZeroDivisionError:
    print "\nDivision by zero."
except TypeError, detail:
    # The detail provides extra information about the exception.
    print "\nSome type mismatch:", detail
except IndexError, detail:
    print "\nSome value is out of range:", detail
except:
    # Except without an exception name catches any exception.
    print "\nSomething else went wrong."

# An else attached to an except block is run if no exception occurrs.
else:
    print "\nThat's odd, nothing went wrong."
