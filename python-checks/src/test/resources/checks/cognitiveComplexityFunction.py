def zero_complexity():
    pass

def if_else_complexity(): # Noncompliant {{Refactor this function to reduce its Cognitive Complexity from 3 to the 0 allowed.}} [[effortToFix=3]]
#   ^^^^^^^^^^^^^^^^^^
    if condition:   # +1
#   ^^<
        pass
    elif condition: # +1
#   ^^^^<
        pass
    else:           # +1
#   ^^^^<
        pass

def decorator_func(some_function):  # Noncompliant [[effortToFix=1]]
    def wrapper():
        if condition:  # +1
            print("before some_function()")
        some_function()

    return wrapper
