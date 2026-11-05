from typing_extensions import Sentinel

MISSING = Sentinel("MISSING")


def equality_comparisons(value):
    if value == MISSING:  # Noncompliant {{Use "is" instead of "==" when comparing with a sentinel value.}}
        #    ^^ 
        pass
    if value != MISSING:  # Noncompliant {{Use "is not" instead of "!=" when comparing with a sentinel value.}}
        #    ^^ 
        pass
