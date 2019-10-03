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

def redundant_continue_in_while(cond):
    while cond:
        continue # Noncompliant

def redundant_continue_in_while_if(cond, p):
    while cond:
        if p:
            continue # Noncompliant

def redundant_continue_in_while_if_else(cond, p):
    while cond:
        if p:
            continue # Noncompliant
        else:
            x = 42

def non_redundant_continue_in_while(cond, p):
    while cond:
        if p:
            continue
        x = 42

def redundant_continue_in_nested_while(cond1, cond2):
    while cond1:
        while cond2:
          continue # Noncompliant

def non_redundant_continue_in_nested_while(cond1, cond2, p):
    while cond1:
        while cond2:
            if p:
                continue
            x = 42

def redundant_continue_in_for(collection):
    for elem in collection:
        continue # Noncompliant

def redundant_continue_in_for_if(collection, p):
    for elem in collection:
        if p:
            continue # Noncompliant

def redundant_continue_in_for_if_else(collection, p):
    for elem in collection:
        if p:
            continue # Noncompliant
        else:
            x = 42

def non_redundant_continue_in_for(collection, p):
    for elem in collection:
        if p:
            continue
        x = 42

def redundant_continue_in_nested_for(collection1, collection2):
    for elem1 in collection1:
        for elem2 in collection2:
          continue # Noncompliant

def non_redundant_continue_in_nested_for(collection1, collection2, p):
    for elem1 in collection1:
        for elem2 in collection2:
            if p:
                continue
            x = 42
