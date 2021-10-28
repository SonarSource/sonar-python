import re


def non_compliant(input):
    re.match(r"(?:)*", input)  # Noncompliant {{Rework this part of the regex to not match the empty string.}}
    #          ^^^^
    re.match(r"(?:)?", input)  # Noncompliant
    re.match(r"(?:)+", input)  # Noncompliant
    re.match(r"()*", input)  # Noncompliant
    re.match(r"()?", input)  # Noncompliant
    re.match(r"()+", input)  # Noncompliant
    re.match(r"xyz|(?:)*", input)  # Noncompliant
    re.match(r"(?:|x)*", input)  # Noncompliant
    re.match(r"(?:x|)*", input)  # Noncompliant
    re.match(r"(?:x|y*)*", input)  # Noncompliant
    re.match(r"(?:x*|y*)*", input)  # Noncompliant
    re.match(r"(?:x?|y*)*", input)  # Noncompliant
    re.match(r"(?:x*)*", input)  # Noncompliant
    re.match(r"(?:x?)*", input)  # Noncompliant
    re.match(r"(?:x*)?", input)  # Noncompliant
    re.match(r"(?:x?)?", input)  # Noncompliant
    re.match(r"(?:x*)+", input)  # Noncompliant
    re.match(r"(?:x?)+", input)  # Noncompliant
    re.match(r"(x*)*", input)  # Noncompliant
    re.match(r"((x*))*", input)  # Noncompliant
    re.match(r"(?:x*y*)*", input)  # Noncompliant
    re.match(r"(?:())*", input)  # Noncompliant
    re.match(r"(?:(?:))*", input)  # Noncompliant
    re.match(r"((?i))*", input)  # Noncompliant
    re.match(r"(())*", input)  # Noncompliant
    re.match(r"(()x*)*", input)  # Noncompliant
    re.match(r"(()|x)*", input)  # Noncompliant
    re.match(r"($)*", input)  # Noncompliant
    re.match(r"(\b)*", input)  # Noncompliant
    re.match(r"((?!x))*", input)  # Noncompliant


def compliant(input):
    re.match(r"x*|", input)
    re.match(r"x*|", input)
    re.match(r"x*", input)
    re.match(r"x?", input)
    re.match(r"(?:x|y)*", input)
    re.match(r"(?:x+)+", input)
    re.match(r"(?:x+)*", input)
    re.match(r"(?:x+)?", input)
    re.match(r"((x+))*", input)
