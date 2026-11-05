MISSING = sentinel("MISSING")
UNSET = sentinel("UNSET")


def equality_comparisons(value):
    if value == MISSING:  # Noncompliant {{Use "is" instead of "==" when comparing with a sentinel value.}}
        #    ^^ 
        pass
    if UNSET == value:  # Noncompliant
        #    ^^
        pass
    if value != MISSING:  # Noncompliant {{Use "is not" instead of "!=" when comparing with a sentinel value.}}
        #    ^^ 
        pass
    if UNSET != value:  # Noncompliant
        #    ^^
        pass


def identity_comparisons(value):
    if value is MISSING:  # OK
        pass
    if UNSET is not value:  # OK
        pass


def other_values(value):
    if value == None:  # OK
        pass
    if value != object():  # OK
        pass


def user_defined_sentinel(value):
    class sentinel:
        pass

    custom = sentinel()
    if value == custom:  # OK
        pass
