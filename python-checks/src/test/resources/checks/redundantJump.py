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
