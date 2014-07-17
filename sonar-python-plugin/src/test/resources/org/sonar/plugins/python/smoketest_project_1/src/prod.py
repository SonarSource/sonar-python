def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)

def unused_function(unused_arg):
    if False:
        unused_code()
    print "nobody calls me :("


# two duplicated blocks

def foo():
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    + ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    + ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    + ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
