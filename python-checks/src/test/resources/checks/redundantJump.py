def redundant_return_at_end():
    x = 1
    return #Noncompliant

def non_redundant_return_at_end():
    x = 1
    return x

def non_redundant_return_in_if(cond):
    if cond:
        return
    x = 42

def redundant_return_in_if(cond):
    if cond:
        return #Noncompliant

def redundant_return_in_if_else(cond):
    if cond:
        x = 10
        return #Noncompliant {{Remove this redundant jump.}}
#       ^^^^^^
    else:
        x = 42

def redundant_return_in_if_elif(cond, cond2):
    if cond:
        x = 10
        return #Noncompliant
    elif cond2:
        x = 42
