#!/usr/bin/python

# Print the contents of the files listed on the command line.

import sys

for fn in sys.argv[1:]:
    try:
        fin = open(fn, 'r')
    except:
        (type, detail) = sys.exc_info()[:2]
        print "\n*** %s: %s: %s ***" % (fn, type, detail)
        continue
    print "\n*** Contents of", fn, "***"

    # Print the file, with line numbers.
    lno = 1
    while 1:
        line = fin.readline()
        if not line: break;
        print '%3d: %-s' % (lno, line[:-1])
        lno = lno + 1
    fin.close()
print
