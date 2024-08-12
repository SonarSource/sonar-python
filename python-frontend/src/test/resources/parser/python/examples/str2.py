#!/usr/bin/python

# Strings have various escapes.
print "Hi\nth\ere,\thow \141\x72\145\x20you?"

# Raw strings ignore them.
print r"Hi\nth\ere,\thow \141\x72\145\x20you?"

print

# Very useful when building file paths on Windows.
badpath = 'C:\that\new\stuff.txt';
print badpath
path = r'C:\that\new\stuff.txt';
print path
