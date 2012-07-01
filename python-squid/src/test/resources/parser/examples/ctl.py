#!/usr/bin/python

# Simple while loop
a = 0
while a < 15:
    print a,		# Trailing comma supresses auto newline.
    if a == 10:
        print "made it to ten!!"
    a = a + 1
