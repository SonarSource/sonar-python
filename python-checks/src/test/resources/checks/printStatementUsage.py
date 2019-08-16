def foo():
    print 1 # Noncompliant {{Replace print statement by built-in function.}}
#   ^^^^^
    print('1')
