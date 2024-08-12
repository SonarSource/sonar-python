#!/usr/bin/python

# The os.path module has operations for manipulating file name paths
# for whaterver system you're running on.
from os.path import *

for fn in [ '/', '/home/bennet', 'path.py', '/var/log/messages', 'bogus' ]:
    print '%-15s' % fn + ':',

    # Don't forget: These operations are os.path.whatever.

    # Is it there?
    if exists(fn):
        print 'exists,',
    else:
        print 'nonexistent'
        print
        continue

    # Absolute path?
    if isabs(fn):
        print 'absolute,',
        print 'directory', dirname(fn)+',', 'base', basename(fn)+','
        print ' ' * 16,
    else:
        print 'relative,',

    # What sort of thing is it?
    if isfile(fn): print 'plain file,',
    elif isdir(fn): print 'directory,',
    else: print 'strange,',

    # Extension.
    print 'extension', splitext(fn)[1]+',',

    # Size
    print getsize(fn), 'bytes.'

    print
