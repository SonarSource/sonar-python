#!/usr/bin/python

#
# Python function arguments may have default values, and can be called
# keyword-style, as in Ada.
def dink(base, middles = [ 'red', 'blue' ], end = '.'):
    'Silly sentence generator.'
    for m in middles:
        print base + m + end

dink('The paint is ')
print
dink('The walls are ', ['painted', 'cracked', 'ugly'], ' like mine.')
print
dink('My car is ', end = ' and broken.')
print
dink('', end = ' with chickens.',
     middles = ['Eating', 'Dancing', 'Watching TV'])
