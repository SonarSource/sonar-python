def hello(): # +1 function
    if expression: # +1 if-statement
        pass
    while expression: # +1 while-statement
        pass
    for i in [0,2]: # +1 for-statement
        pass
    return # +1 return-statement

expression = a or b  # +1
expression = a and b # +1
expression = a if b else c # +1

try:
    raise "message" # +1 raise-statement
except: # +1 except-clause
    pass
