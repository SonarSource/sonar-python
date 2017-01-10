def hello(): # +1 function
    while expression: # +1 while-statement
        pass
    for i in [0,2]: # +1 for-statement
        if expression: # +1 if-statement
            continue # +0
        else:
            break # +0
        pass
    return # +0

expression = a or b  # +1
expression = a and b # +1
expression = a if b else c # +1
expression = lambda x: x*2 # +0

try:
    raise "message" # +0
except: # +0
    pass
