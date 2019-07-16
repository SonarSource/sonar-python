def getS():
    return "%s"

def foo():
    exec 'print 1' # Noncompliant {{Do not use exec statement.}}
#   ^^^^
    exec('print 1')
    exec ('print 1')
    exec "%s %s" % ("a", "b") # Noncompliant {{Do not use exec statement.}}
    exec getS() % "a" # Noncompliant {{Do not use exec statement.}}
