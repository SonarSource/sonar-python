import re


def non_compliant(input):
    re.match("(mandatory||optional)", input)             # Noncompliant {{Remove this empty alternative.}}
#                        ^
    re.match("|mandatory|-optional", input)              # Noncompliant
#             ^

    re.match("mandatory|-optional|", input)              # Noncompliant
#                                ^
    # Noncompliant@+3
    re.match('''(
    \'[^\']*(\'|$)|     # - a string that starts with a quote, up until the next quote or the end of the string
    |                   # or
    \S                  # - a non-whitespace character
    )''', input, re.X)

    re.match("|mandatory|-optional", input)              # Noncompliant
    re.match("(mandatory|(|O|o|)ptional|)", input)       # Noncompliant
    re.match("(|mandatory|optional)?", input)            # Noncompliant
#              ^
    re.match("mandatory(-optional|){2}", input)          # Noncompliant
#                                ^


def compliant(input):
    re.match("(mandatory|optional|)", input)
    re.match("mandatory(-optional|)", input)
    re.match("mandatory(|-optional)", input)
    re.match("mandatory(|-optional)", input)
    re.match("mandatory(-optional|)", input)
    re.match("(mandatory(|-optional))?", input)
