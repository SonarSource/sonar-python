#!/usr/bin/python

# Script to copy standard input to standard output, one line at a time.

# This gets various items to interfaces with the OS, including the
# standard input stream.
import sys

# Readline is a method of stdin, which is in the standard object sys.
# It returns the empty string on EOF.
line = sys.stdin.readline()

# The string line works as the while test.  As several other scripting
# languages, the empty string is treated as false, other strings are treated
# as true.
while line:
    # Print the line read.  Since readline leaves the terminating newline,
    # a slice is used to print all characters in the string but the last.
    # Otherwise, each input line would be output with two line terminators.
    print line[:-1]

    # Next line.
    line = sys.stdin.readline()
