import re

def non_compliant():
    re.match("\\01b", "\x01b") # Noncompliant {{Consider replacing this octal escape sequence with a Unicode or hexadecimal sequence instead.}}
#             ^^^^ 0
    re.match("(x)\1", "x\x01")  # Noncompliant
    re.match("\\076b", ">b") # Noncompliant
    re.match("\\0761", ">1") # Noncompliant
    re.match("\\276", "\xBE") # Noncompliant
    re.match("\\276b", "\xBEb") # Noncompliant
    re.match("\\2760", "\xBE0") # Noncompliant


def non_compliant_raw_strings():
    re.match(r"\01b", "\x01b") # Noncompliant  {{Consider replacing this octal escape sequence with a Unicode or hexadecimal sequence instead.}}
#              ^^^ 0
    re.match(r"\076b", ">b") # Noncompliant
    re.match(r"\0761", ">1") # Noncompliant
    re.match(r"\101", "A") # Noncompliant
    re.match(r"\101B", "AB") # Noncompliant
    re.match(r"\1011", "A1") # Noncompliant

def compliant():
    re.match("\\x01", "\x01")  # Hexadecimal escape sequence
    re.match("(x)\\1", "xx")  # Back reference
    re.match(r"(x)\1", "xx")  # Back reference
    re.compile("\\4760") # Back reference
