#!/usr/bin/python

# This contains misc system services.
import sys

# Delete operation.
try:
    fred = [ 4, 5, 11, 43 ]
    print 'fred =', fred
    del fred[2]
    print 'fred =', fred
    del fred
    print 'fred =', fred
    print "Doesn't get here!"
except NameError, x:
    print "*** Name", x, "undefined ***"

fred = 'Nimrod'
print 'fred =', fred
