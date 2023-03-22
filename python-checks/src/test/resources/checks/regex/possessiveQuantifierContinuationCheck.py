import re

def foo():
    match1 = re.match(r"a++abc", "aaaabc", re.DOTALL) # Noncompliant
#                          ^

    match2 = re.match(r"\d*+[02468]", "1234", re.DOTALL) # Noncompliant {{Change this impossible to match sub-pattern that conflicts with the previous possessive quantifier.}}
    #                       ^^^^^^^
    match3 = re.match(r"aa++bc", "aaaabc", re.DOTALL) # Compliant, for example it can match "aaaabc"
    match4 = re.match(r"\d*+(?<=[02468])", "1234", re.DOTALL) # Compliant, for example, it can match an even number like "1234"

    re.match(r"a+abc", "aaaabc", re.DOTALL)
    re.match(r"a+?abc", "aaaabc", re.DOTALL)
    re.match(r"a++abc", "aaaabc", re.DOTALL) # Noncompliant {{Change this impossible to match sub-pattern that conflicts with the previous possessive quantifier.}}
    re.match(r"\d*+[02468]", "aaaabc", re.DOTALL) # Noncompliant
    re.match(r"(\d)*+([02468])", "aaaabc", re.DOTALL) # Noncompliant
    re.match(r"\d++(?:[eE][+-]?\d++)?[fFdD]?", "aaaabc", re.DOTALL)
    re.match(r"a*+\s", "aaaabc", re.DOTALL)
    re.match("[+-]?(?:NaN|Infinity|(?:\\d++(?:\\.\\d*+)?|\\.\\d++)(?:[eE][+-]?\\d++)?[fFdD]?|0[xX](?:\\p{XDigit}++(?:\\.\\p{XDigit}*+)?|\\.\\p{XDigit}++)[pP][+-]?\\d++[fFdD]?)", "aaaabc", re.DOTALL)
    re.match("aa++bc", "aaaabc", re.DOTALL)
    re.match(r"\d*+(?<=[02468])", "aaaabc", re.DOTALL)
    re.match(r"(xx++)+x", "aaaabc", re.DOTALL) # Noncompliant
    re.match(r"(bx++)+x", "aaaabc", re.DOTALL) # Noncompliant
    re.match(r"(?:xx++)+x", "aaaabc", re.DOTALL) # Noncompliant
    re.match(r"(xx++)x", "aaaabc", re.DOTALL) # Noncompliant
    re.match(r".*+\w", "aaaabc", re.DOTALL) # FN
    re.match(r".*+\w+", "aaaabc", re.DOTALL) # FN
    re.match(r"(a|b|c)*+(a|b)", "aaaabc", re.DOTALL) # Noncompliant
    re.match(r"(:[0-9])?+(:[0-9])?+", "aaaabc", re.DOTALL)
