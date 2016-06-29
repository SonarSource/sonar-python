for y in range(10):
    if x > y:
        if x > 1:
            pass

for x in range(10):
    while x:
        for y in range(10):
            if x > y:
                if x > 1:  # Noncompliant [[secondary=-4,-3,-2,-1]] {{Refactor this code to not nest more than 4 "if", "for", "while", "try" and "with" statements.}}
#               ^^
                    if y > 10:
                        pass
