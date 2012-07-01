#!/usr/bin/python

# List the account and human names from the passwd file in sorted order.  
# If there is no human name, or it equals the userid, then the name is
# printed as [ none ].   The program reads from the file given the command
# line, if any,  else it reads from /etc/passwd.  If the file open fails, the
# program will die on an exception.

# The from form of of import adds name to the symbol table of the
# importing program, so that the names may be refered to without
# the qualified module name.
from sys import argv
from string import *

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

# Loop through each input line.
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

# Sort the names, then print.
ids = iddict.keys()
ids.sort()
for userid in ids:
    # Get the human name.
    human = iddict[userid]

    # Pad the account name.
    if len(userid) < 12:
        userid = userid + ' ' * (12 - len(userid))

    # Spit it out.
    print userid + human
