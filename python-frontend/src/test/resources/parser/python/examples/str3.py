#!/usr/bin/python

# Use brackets for string subscripting and substrings.
bozon = 'Cheer for Friday'
#	 0123456789012345

# Index at zero.
print bozon[0], bozon[1], bozon[15]

# You may not use [] on the left of an assignment, however.

# Use the colon to describe ranges.  The last character is not part of the
# range.
print bozon[0:5] + ", " + bozon[0:6] + bozon[10:16] + '!'

# Defaults to first and last.
print bozon[:5] + ", " + bozon[:6] + bozon[10:] + '!'

# The * operator repeats strings (like x in perl).
print bozon[:6] * 3 + '!'

# Negatives index back from the right, the rightmost character being -1.
print bozon[-1] + ' ' + bozon[-10:-6]
