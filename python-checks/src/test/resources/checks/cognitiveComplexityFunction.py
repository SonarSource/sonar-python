def zero_complexity():
    pass

def if_else_complexity(): # Noncompliant [[effortToFix=3;secondary=+2,+4,+6]] {{Refactor this function to reduce its Cognitive Complexity from 3 to the 0 allowed.}}
#   ^^^^^^^^^^^^^^^^^^
    if condition:   # +1
        pass
    elif condition: # +1
        pass
    else:           # +1
        pass


def else_nesting():    # Noncompliant [[effortToFix=4]]
    if condition:    # +1
        pass
    else:            # +1 (nesting level +1)
        if condition:# +2
            pass


def else_nested():    # Noncompliant [[effortToFix=4]]
    if condition:     # +1 (nesting level +1)
        if condition: # +2
            pass
        else:         # +1
            pass


def if_nested():    # Noncompliant [[effortToFix=6]]
    if condition:  # +1 (nesting level +1)
        if condition:  # +2 (nesting level +1)
            if condition: # +3
                pass


def elif_nesting():    # Noncompliant [[effortToFix=4]]
    if condition:      # +1
        pass
    elif condition:    # +1 (nesting level +1)
        if condition:  # +2
            pass


def while_complexity():    # Noncompliant [[effortToFix=4]]
    while condition:  # +1 (nesting level +1)
        if condition: # +2
            pass
        else:         # +1
            pass;


def while_with_else_complexity():    # Noncompliant [[effortToFix=4]]
    while condition:  # +1
        pass
    else:             # +1 (nesting level +1)
        if condition: # +2
            pass


def for_complexity():    # Noncompliant [[effortToFix=4]]
    for x in "foo":  # +1 (nesting level +1)
        if condition:# +2
            pass
        else:        # +1
            pass;


def for_with_else_complexity():    # Noncompliant [[effortToFix=4]]
    for x in "foo":   # +1
        pass
    else:             # +1 (nesting level +1)
        if condition: # +2
            pass


def try_except():    # Noncompliant [[effortToFix=11]]
    try:
        if condition: # +1
            pass
    except SomeError1: # +1 (nesting level +1)
        if condition: # +2
            pass
    except SomeError2: # +1 (nesting level +1)
        if condition: # +2
            pass
    except (SomeError3, SomeError4): # +1 (nesting level +1)
        if condition: # +2
            pass
    finally:
        if condition: # +1
            pass


def try_finally():    # Noncompliant [[effortToFix=2]]
    try:
        if condition: # +1
            pass
    finally:
        if condition: # +1
            pass


def try_else():    # Noncompliant [[effortToFix=4]]
    try:
        pass
    except SomeError: # +1
        pass
    else:             # +1 (nesting level +1)
        if condition: # +2
            pass

def jump_statements():   # Noncompliant [[effortToFix=6]]
    if condition:        # +1
        return
    elif condition:      # +1
        return 42

    while condition:     # +1 (nesting level +1)
        if condition:    # +2
            break
        elif condition:  # +1
            continue

# TODO recursion +1
def recursion():  # Noncompliant [[effortToFix=2]]
    if condition:
        return 42
    else:
        return recursion()


def nesting_func():  # Noncompliant [[effortToFix=3]]
    if condition:  # +1
        pass
    def nested_func(): # +0 (nesting level +1)
        if condition:  # +2
            pass

def decorator_func(some_function):  # Noncompliant [[effortToFix=1]]
    def wrapper():
        if condition:  # +1
            print("before some_function()")
        some_function()

    return wrapper

def complex_decorator_func(some_function):  # Noncompliant [[effortToFix=3]]
    def wrapper(): # +0 (nesting level +1)
        if condition:  # +2
            print("before some_function()")
        some_function()

    return wrapper if condition else None # +1

def and_or():  # Noncompliant [[effortToFix=7]]
    foo(1 and 2 and 3 and 4) # +1
    foo(1 or 2 or 3 or 4)    # +1
    foo(1 and 2 or 3 or 4)   # +2
    foo(1 and 2 or 3 and 4)  # +3

class A:
    def method(self):  # Noncompliant [[effortToFix=1]]
        if condition:  # +1
            class B:
                pass

def conditional_expression(): # Noncompliant [[effortToFix=1]]
    return true_value if condition else false_value

# TODO nested conditional expressions
def nested_conditional_expression_false(): # Noncompliant [[effortToFix=2]]
    x = true_value1 if condition1 else (true_value2 if condition2 else false_value) # should be +3

# TODO nested conditional expressions
def nested_conditional_expression_true(): # Noncompliant [[effortToFix=2]]
    y = (true_value1 if condition1 else false_value1) if condition2 else false_value2 # should be +3
