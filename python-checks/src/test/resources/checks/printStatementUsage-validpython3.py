import sys
if (2 > 3):
    print >>sys.stderr, ("hello") #Noncompliant
#   ^^^^^
    print + sys.stderr, ("hello")
    abcde  >>sys.stderr, ("hello")
    print("hello", file=sys.stderr)

    print >> f #Noncompliant
    x = print >> f

    print #Noncompliant
    print()
    myprint
