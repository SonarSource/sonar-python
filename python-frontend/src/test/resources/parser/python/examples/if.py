#!/usr/bin/python
#
# Python program to generate a random number with commentary.
#

# Import is much like Java's.  This gets the random number generator.
import random

# Generate a random integer in the range 10 to 49.
i = random.randrange(10,50)
print 'Your number is', i

# Carefully analyze the number for important properties.
if i < 20:
    print "That is less than 20."
    if i % 3 == 0:
        print "It is divisible by 3."
elif i == 20:
    print "That is exactly twenty.  How nice for you."
else:
    if i % 2 == 1:
        print "That is an odd number."
    else:
        print "That is twice", i / 2, '.'
    print "Wow! That's more than 20!"

