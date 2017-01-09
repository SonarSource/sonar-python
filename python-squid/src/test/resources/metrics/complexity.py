def hello(): # +1 function
    if expression: # +1 if-statement
        pass
    while expression: # +1 while-statement
        pass
    for i in [0,2]: # +1 for-statement
        pass
    return # +0 return-statement

expression = a or b  # +1
expression = a and b # +1
expression = a if b else c # +1

try:
    raise "message" # +0 raise-statement
except: # +0 except-clause
    pass
