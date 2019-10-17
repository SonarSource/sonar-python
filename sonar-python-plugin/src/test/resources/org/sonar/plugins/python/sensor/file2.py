foo(); bar(); # Noncompliant "OneStatementPerLine"

for x in range(10):
    while x:
        for y in range(10):
            if x > y:
                if x > 1: # Noncompliant "S134"
                        pass
