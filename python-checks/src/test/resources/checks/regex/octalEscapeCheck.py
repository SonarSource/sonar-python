import re

def non_compliant():
    re.compile("\\01b") # Noncompliant {{Consider replacing this octal escape sequence with a Unicode or hexadecimal sequence instead.}}
#               ^^^^ 0
    re.compile("\\076b") # Noncompliant {{Consider replacing this octal escape sequence with a Unicode or hexadecimal sequence instead.}}
#               ^^^^^ 0
    re.compile("\\276b") # Noncompliant {{Consider replacing this octal escape sequence with a Unicode or hexadecimal sequence instead.}}
#               ^^^^^ 0
    re.compile("\\2760") # Noncompliant {{Consider replacing this octal escape sequence with a Unicode or hexadecimal sequence instead.}}
#               ^^^^^ 0

def compliant():
    re.compile("\\x01")
    re.compile("(x)\\1")
    re.compile(r"(x)\1")
    re.compile("\\4760") # Back reference
