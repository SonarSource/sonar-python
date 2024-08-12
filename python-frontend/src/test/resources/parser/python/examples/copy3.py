#!/usr/bin/python

# Script to copy standard input to standard output using the readlines
# operation which (at least virtually) reads the entire file.

import sys

# Loop through each input line.
for line in sys.stdin.readlines():
    # Print the line read.
    print line[:-1]
