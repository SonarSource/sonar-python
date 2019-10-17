#!/usr/bin/python

# Dictionaries map keys to values.

fred = { 'mike': 456, 'bill': 399, 'sarah': 521 }

# Subscripts.
try:
    print fred
    print fred['bill']
    print fred['nora']
    print "Won't see this!"
except KeyError, rest:
    print "Lookup failed:", rest
print

# Entries can be added, udated, or deleted.
fred['bill'] = 'Sopwith Camel'
fred['wilma'] = 2233
del fred['mike']
print fred
print

# Get all the keys.
print fred.keys()
for k in fred.keys():
    print k, "=>", fred[k]
print

# Test for presence of a key.
for t in [ 'zingo', 'sarah', 'bill', 'wilma' ]:
    print t,
    if fred.has_key(t):
        print '=>', fred[t]
    else:
        print 'is not present.'

