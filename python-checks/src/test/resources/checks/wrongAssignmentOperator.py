target = -5
num = 3

target =+ num # Noncompliant {{Was += meant instead?}}
#      ^^
target =- num # Noncompliant {{Was -= meant instead?}}
#      ^^
target=+ num # Noncompliant
target =+num # Noncompliant
target += num
# Noncompliant@+1 {{Was += meant instead?}}
target=+\
        num
#     ^^@-1

target=+num # Compliant, no spaces
target = num =+ num # Compliant, an augmented assignment would not be valid here
