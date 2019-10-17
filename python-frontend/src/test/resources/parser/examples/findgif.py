#!/usr/bin/python

# Use the standard find method to look for GIF files.
import sys, find

if len(sys.argv) > 1:
    dirs = sys.argv[1:]
else:
    dirs = [ '.' ]

# Go for it.
for dir in dirs:
    files = find.find('*.gif', dir)
    if files:
        print "For", dir + ':'
        for fn in files:
            print " ", fn
    else:
        print "For", dir + ': None'

