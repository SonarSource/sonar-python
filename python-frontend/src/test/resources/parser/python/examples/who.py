#!/usr/bin/python

# List the account and human names from an /etc/passwd-style
# file on standard input.  If there is no human name, or it equals the
# userid, then the name is printed as [ none ]

import sys, string

# Loop through each input line.
for line in sys.stdin.readlines():
    # Lines starting with # are comments.  Clean leading spaces, and
    # skip comments.
    line = string.lstrip(line)
    if line == '' or line[0] == '#':
        continue

    # Split the line by the : delimeter, extract the appropriate fields,
    # and get rid of any leading or trailing blanks.
    parts = string.split(line, ':')
    userid = string.strip(parts[0])
    name = string.strip(parts[4])

    # Trim the contents of the name following the first comma, if any.
    compos = string.find(name, ',')
    if compos != -1:
        name = name[0:compos]

    # If there is no human name, or if equals the login name, say [ none ]
    if name == '' or name == userid:
        name = '[ none ]'

    # Pad the account name.
    if len(userid) < 12:
        userid = userid + ' ' * (12 - len(userid))

    # Spit it out.
    print userid + name
