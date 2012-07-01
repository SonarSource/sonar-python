#!/usr/bin/python

# Script to copy standard input to standard output, one line at a time,
# now using a break.

import sys

# Loop until terminated by the break statement.
while 1:
    # Get the line, exit if none.
    line = sys.stdin.readline()
    if not line:
        break

    # Print the line read.
    print line[:-1]
