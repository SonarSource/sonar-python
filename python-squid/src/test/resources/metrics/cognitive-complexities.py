def zero_complexity(): # =0
    pass

def if_else_complexity(): # =3
    if condition:         # +1
        pass
    elif condition:       # +1
        pass
    else:                 # +1
        pass


def else_nesting():   # =4
    if condition:     # +1
        pass
    else:             # +1
        if condition: # +2 (incl 1 for nesting)
            pass


def else_nested():    # =4
    if condition:     # +1
        if condition: # +2 (incl 1 for nesting)
            pass
        else:         # +1
            pass


def if_nested():          # =6
    if condition:         # +1
        if condition:     # +2 (incl 1 for nesting)
            if condition: # +3 (incl 2 for nesting)
                pass


def elif_nesting():    # =4
    if condition:      # +1
        pass
    elif condition:    # +1
        if condition:  # +2 (incl 1 for nesting)
            pass


def while_complexity(): # =4
    while condition:    # +1
        if condition:   # +2 (incl 1 for nesting)
            pass
        else:           # +1
            pass;


def while_with_else_complexity(): # =4
    while condition:              # +1
        pass
    else:                         # +1
        if condition:             # +2 (incl 1 for nesting)
            pass


def for_complexity():    # =4
    for x in "foo":      # +1
        if condition:    # +2 (incl 1 for nesting)
            pass
        else:            # +1
            pass;


def for_with_else_complexity(): # =4
    for x in "foo":             # +1
        pass
    else:                       # +1
        if condition:           # +2 (incl 1 for nesting)
            pass


def try_except():                    # =11
    try:
        if condition:                # +1
            pass
    except SomeError1:               # +1
        if condition:                # +2 (incl 1 for nesting)
            pass
    except SomeError2:               # +1
        if condition:                # +2 (incl 1 for nesting)
            pass
    except (SomeError3, SomeError4): # +1
        if condition:                # +2 (incl 1 for nesting)
            pass
    finally:
        if condition:                # +1
            pass


def try_finally():    # =2
    try:
        if condition: # +1
            pass
    finally:
        if condition: # +1
            pass


def try_else():       # =4
    try:
        pass
    except SomeError: # +1
        pass
    else:             # +1
        if condition: # +2 (incl 1 for nesting)
            pass

def jump_statements():   # =6
    if condition:        # +1
        return
    elif condition:      # +1
        return 42

    while condition:     # +1
        if condition:    # +2 (incl 1 for nesting)
            break
        elif condition:  # +1
            continue

# TODO recursion +1
def recursion():  # =2
    if condition: # +1
        return 42
    else:         # +1
        return recursion()

def nesting_func():    # =3
    if condition:      # +1
        pass
    def nested_func(): # nesting level +1
        if condition:  # +2 (incl 1 for nesting)
            pass

def decorator_func(some_function):          # =1
    def wrapper():                          # nesting level +0
        if condition:                       # +1
            print("before some_function()")
        some_function()

    return wrapper

def complex_decorator_func(some_function):  # =3
    def wrapper():                          # nesting level +1
        if condition:                       # +2 (incl 1 for nesting)
            print("before some_function()")
        some_function()

    return wrapper if condition else None   # +1

def and_or():                # =7
    foo(1 and 2 and 3 and 4) # +1
    foo(1 or 2 or 3 or 4)    # +1
    foo(1 and 2 or 3 or 4)   # +1 +1
    foo(1 and 2 or 3 and 4)  # +1 +1 +1

class A:
    def method(self):  # =1
        if condition:  # +1
            class B:
                pass

def conditional_expression():                       # =1
    return true_value if condition else false_value # +1

def nested_conditional_expression_false():                                          # =3
    x = true_value1 if condition1 else (true_value2 if condition2 else false_value) # +1 +2 (incl 1 for nesting)

def nested_conditional_expression_true():                                             # =3
    y = (true_value1 if condition1 else false_value1) if condition2 else false_value2 # +1 +2 (incl 1 for nesting)

def not_complex(some_function):  # =0
    def nested_not_complex():    # nesting level +1
        some_function()
    some_function()
    return nested_not_complex

# Code outside functions
if condition:         # +1
    pass
elif condition:       # +1
    pass
else:                 # +1
    pass

def function_with_nested_class(a):  # =4
    def f(b):
        return 1 if b else 2        # +2 (incl 1 for nesting)

    class A:
        # class reset nesting level
        def m(self, b):
            return 1 if b else 2    # +1

    return 1 if b else 2            # +1
