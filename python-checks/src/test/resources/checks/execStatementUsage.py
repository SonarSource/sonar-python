def foo():
    exec 'print 1' # Noncompliant {{Do not use exec statement.}}
#   ^^^^
    exec('print 1')
