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

def two_redundant_return_if_else(cond):
    if cond:
        x = 10
        return #Noncompliant
    else:
        x = 42
        return #Noncompliant

def two_non_redundant_return_if_else(cond):
    if cond:
        x = 10
        return
    else:
        x = 42
        return
    x = 2

def two_redundant_return_if_elif(cond1, cond2):
    if cond1:
        x = 10
        return #Noncompliant
    elif cond2:
        x = 42
        return #Noncompliant

def two_non_redundant_return_if_elif(cond1, cond2):
    if cond1:
        x = 10
        return
    elif cond2:
        x = 42
        return
    x = 2


def three_redundant_return_if_elif_else(cond1, cond2):
    if cond1:
        x = 10
        return #Noncompliant
    elif cond2:
        x = 42
        return #Noncompliant
    else:
        x = 11
        return #Noncompliant

def three_non_redundant_return_if_elif_else(cond1, cond2):
    if cond1:
        x = 10
        return
    elif cond2:
        x = 42
        return
    else:
        x = 11
        return
    x = 2
