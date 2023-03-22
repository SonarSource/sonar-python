import re

def non_compliant():
    re.compile("\\01b") # Noncompliant {{Replace with hexadecimal code}}
#               ^^^^ 0
    re.compile("\\076b") # Noncompliant {{Replace with hexadecimal code}}
#               ^^^^^ 0
    re.compile("\\276b") # Noncompliant {{Replace with hexadecimal code}}
#               ^^^^^ 0
def compliant():
    re.compile("\\x01")
