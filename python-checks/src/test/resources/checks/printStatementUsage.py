import sys

def getS():
    return "%s"

def foo():
    print 1 # Noncompliant {{Replace print statement by built-in function.}}
#   ^^^^^
    print('1')
    print ('1')
    print "%s %s" % ("a", "b") # Noncompliant {{Replace print statement by built-in function.}}
    print getS() % "a" # Noncompliant {{Replace print statement by built-in function.}}

print >>sys.stderr, ("fatal error") # Noncompliant
print >>sys.stderr, "fatal error" # Noncompliant

print # Noncompliant {{Replace print statement by built-in function.}}
