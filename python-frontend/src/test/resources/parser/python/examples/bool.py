#!/usr/bin/python

# Script to demonstrate Python comparison and boolean operators.
import random

# Some relationals.  Relationals in Python can be chained, and are
# interpreted with an implicit and.
c = -2
for a in range(1,4):
    c = c + 4
    for b in range(1,4):
        print '(' + str(a), '<', str(b) + ') ==', a < b, '  ',
        print '(' + str(a), '>=', b, '>', str(c) + ') ==', a >= b > c, '  ',
        print '(' + str(a), '==', b, '==', str(c) + ') ==', \
              a == b == c, '  ',
        print '(' + str(a), '!=', b, '!=', str(c) + ') ==', a != b != c
        c = c - 1
print

# Some boolean operations on comparisons.  You have to spell these out
# Pascal- or Ada- style.  None of this && and || stuff.  (Appeals to
# shiftless typists.)
c = -1
for a in range(0,3):
    c = c + 5
    for b in range(0,3):
        print '(' + str(a), '==', b, 'or', a, '==', c, 'and', b, '<', \
              str(c) + ') ==', a == b or a == c and b < c, '  ',
        print '(not', a, '<', str(b) + ') == ', not a < b, '  '
        c = c - 2
print

# When and or or returns true, it returns the second argument.
c = -1
for a in [0, 3, 4]:
    c = c + 2
    for b in [-2, 0, 5]:
        print '(' + str(a), 'and', b, 'or', str(c) + ') == ',\
              a and b or c, '  ',
        print '(' + str(a), 'or', b, 'and', str(c) + ') == ',\
              a or b and c
        c = c - 1
print

# Don't forget the very useful in operator.  This works on most (all?) of the
# built-in data structures, including strings.
some = [2,4,7]
for a in range(1,5):
    if a in some:
        print a, 'is',
    else:
        print a, 'is not',
    print 'in', some
