import re


def non_compliant(input):
    re.match(r'(', input)  # Noncompliant {{Fix the syntax error inside this regex.}}
    #           ^
    #           ^@-1< {{Expected ')', but found the end of the regex}}
    re.match(r'$[a-z^', input)  # Noncompliant {{Fix the syntax error inside this regex.}}
    #                ^
    #                ^@-1< {{Expected ']', but found the end of the regex}}
    re.match(r'(\w+-(\d+)', input)  # Noncompliant {{Fix the syntax error inside this regex.}}
    #                    ^
    #                    ^@-1< {{Expected ')', but found the end of the regex}}


def compliant(input):
    re.match(r'\(\[', input)


def unsupported_feature(input):
    # atomic group
    re.match(r'(?>x)', input)  # Noncompliant
    # possessive quantifier
    re.match(r'x*+', input)  # Noncompliant


def false_positives():
    re.compile(r"\s*([ACGT])\s*[[]*[|]*\s*([0-9.\s]+)\s*[]]*\s*")  # Noncompliant
    re.compile(r'^\s+\[([\s*[0-9]*)\] ([a-zA-Z0-9_]*)')  # Noncompliant
    re.compile(r'([^,[\]]*)(\[([^\]]+)\])?$')  # Noncompliant
