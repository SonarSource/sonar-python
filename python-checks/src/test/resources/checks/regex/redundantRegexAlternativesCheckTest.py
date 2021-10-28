import re


def non_compliant(input):
    re.match(r"a|[ab]", input)  # Noncompliant {{Remove or rework this redundant alternative.}}
    #          ^ ^^^^< {{Alternative to keep}}
    re.match(r".|a", input)  # Noncompliant
    re.match(r"a|.", input)  # Noncompliant
    re.match(r"(.)|(a)", input)  # Noncompliant
    re.match(r"a|b|bc?", input)  # Noncompliant
    re.match(r"a|b|bc*", input)  # Noncompliant
    re.match(r"a|b|bb*", input)  # Noncompliant
    re.match(r"a|b|a|b|a|b|a|b", input)  # Noncompliant 2
    re.match(r"[1-2]|[1-4]|[1-8]|[1-3]", input)  # Noncompliant
    re.match(r"1|[1-2]", input)  # Noncompliant


def compliant(input):
    re.match(r"(a)|(.)", input)  # Compliant
    re.match(r"a|(.)", input)  # Compliant
    re.match(r"(a)|.", input)  # Compliant
    re.match(r"a|b|bc+", input)  # Compliant
    re.match(r"|a", input)  # Compliant
    re.match(r"[ab]", input)  # Compliant
    re.match(r".*", input)  # Compliant
