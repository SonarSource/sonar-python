#!/usr/bin/python

# Strings in single quotes.
s1 = 'I\'m single'

# Double quotes are the same.  You just have to end with the same character
# you started with, and you don't have to escape the other one.
s2 = "I'm double double"

# You can create multi-line strings with triple quotes (triple double quotes
# work, too.)  The newlines stay in the string.
s3 = '''I'm very long-winded and I really need
to take up more than one line.  That way I can say all the very
`important' things which I must tell you.  Strings like me are useful
when you must print a long set of instructions, etc.'''

# String literals may be concatinated by a space.
s4 = 'left' "right" 'left'

# Any string expression may be concatinated by a + (Java style).
s5 = s1 + "\n" + s2

print s5 + '\n' + s3, "\n", s4

print 's5 has', len(s5), 'characters'
