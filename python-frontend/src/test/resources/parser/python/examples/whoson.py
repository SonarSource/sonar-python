#!/usr/bin/python

# List the account and human names for each person who's logged on.
# The information is taken from the password file, which is
# read and loaded first.  The command line argument (if given) is
# used in place of /etc/passwd.
from sys import argv
from string import *
import os

# Dictionary of entries.
iddict = { }

# Get the file name.  Note that argv[0] contains the name of the script,
# so we're getting argv[1], if there is one.
infile = '/etc/passwd'
if len(argv) > 1:
    infile = argv[1]

# Attempt to open the file.  Will just let the program die on the exception
# if the open fails.
fin = open(infile, 'r')

# Read the password file and load the information.
for line in fin.readlines():
    # Lines starting with # are comments.  Clean leading spaces, and
    # skip comments.
    line = lstrip(line)
    if line == '' or line[0] == '#':
        continue

    # Split the line by the : delimeter, extract the appropriate fields,
    # and get rid of any leading or trailing blanks.
    parts = split(line, ':')
    userid = strip(parts[0])
    name = strip(parts[4])

    # Trim the contents of the name following the first comma, if any.
    compos = find(name, ',')
    if compos != -1:
        name = name[0:compos]

    # If there is no human name, or if equals the login name, say [ none ]
    if name == '' or name == userid:
        name = '[ none ]'

    # Enter into the dictionary list.
    iddict[userid] = name

# Run who to see who is on, and print those users.  When printed, take
# them out of the list so each user printed only once.
for line in os.popen('who').readlines():
    user = split(line)[0]
    if iddict.has_key(user):
        print '%-14s %s' % (user + ':', iddict[user])
        del iddict[user]
