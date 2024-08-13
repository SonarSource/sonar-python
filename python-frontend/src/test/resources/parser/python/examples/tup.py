#!/usr/bin/python

# Tuples are immutable sequences, much like lists.
a = 5, 9, 'frank', 33
b = ('this', 'that', 'the other')
print "A:", a, b

# They can be concatinated, subscripted, and sliced.
c = a + b
print "B:", c
print "C:", a[2], c[3:]

# They can be taken apart, but the sizes must match!
w, x, y = b
print "D:", w, x, y
try:
    print "E", len(c)
    (p, q, s, f) = c
    print "F:", p, q, s, f
except ValueError, descr:
    print "*** That won't work:", descr, "***"
(p, q, s, f) = c[:4]
print "G:", p, q, s, f

# Sub-tuples are allowed.
mrbig = (5, 17, 4, ('mac', 'alex', 'sally'), 888, b)
print "H:", mrbig

# Empty tuples are allowed, and singleton tuples are ugly.
mt = ()
singleton = (5,)
print "I:", mt, singleton

# Tuples are immutable.
try:
    fred = 5, 9, 22
    fred[1] = 3
    print "Won't see this."
except TypeError, descr:
    print "*** That won't work:", descr, "***"

# Tuples may contain mutable objects.
fred = (5, 9, [3, 4, 7])
print "J:", fred
fred[2][1] = 'cow'
print "K:", fred
