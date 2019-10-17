#!/usr/bin/python

# A few more loops.

# Powers of 2 (for no obvious reason)
power = 1
for y in range(0,21):
    print "2 to the", y, "is", power
    power = 2 * power

# Scanning a list.
fred = ['And', 'now', 'for', 'something', 'completely', 'different.'];
for i in range(0,len(fred)):
    print i, fred[i]
