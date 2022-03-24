import re


def non_compliant(input):
    re.match(r"(?:number)", input)  # Noncompliant {{Unwrap this unnecessarily grouped subpattern.}}
    #          ^^^^^^^^^^
    re.match(r"(?:number)\d{2}", input)  # Noncompliant
    re.match(r"(?:number(?:two){2})", input)  # Noncompliant
    #          ^^^^^^^^^^^^^^^^^^^^
    re.match(r"(?:number(?:two)){2}", input)  # Noncompliant
    #                   ^^^^^^^
    re.match(r"foo(?:number)bar", input)  # Noncompliant
    re.match(r"(?:)", input)  # Noncompliant


def compliant(input):
    re.match(r"(?:number)?+", input)
    re.match(r"number\d{2}", input)
    re.match(r"(?:number)?\d{2}", input)
    re.match(r"(?:number|string)", input)
