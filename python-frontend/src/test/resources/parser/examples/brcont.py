#!/usr/bin/python
#
# This example was shamelessly stolen from section 4 of the Python tutorial at
# http://www.python.org/doc/current/tut/tut.html.
#
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print n, 'equals', x, '*', n/x
            break # Break in python is like the one in C.
    else:
        # The else on a loop is executed after the loop exits normally, but
        # not when it exits prematurely with a break.
        print n, 'is a prime number'
