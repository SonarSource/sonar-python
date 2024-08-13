#!/usr/bin/python

import random

# Using the back ticks, which convert to string.
for n in range(0,5):
    a = random.randrange(0, 101)
    b = random.randrange(0, 201)
    print `a` + '+' + `b`, '=', `a+b`
print

# Using the % operator, similar to printf.
for n in range(0,5):
    a = random.randrange(0, 101)
    b = random.randrange(0, 201)
    print '%d+%d = %d' % (a, b, a + b)
print

# % allows field sizes as well.
for n in range(0,5):
    a = random.randrange(-100, 101)
    b = random.randrange(-50, 201)
    print '%4d + %4d = %4d' % (a, b, a + b)
print

# Some other formatting.
dribble = { 'onion' : 1.4,
            'squash' : 2.02,
            'carrots' : 1.0,
            'pixie toenails' : 43.22,
            'lemon drops' : .75 }
for n in dribble.keys():
    print '%-15s %6.2f' % (n + ':', dribble[n])
