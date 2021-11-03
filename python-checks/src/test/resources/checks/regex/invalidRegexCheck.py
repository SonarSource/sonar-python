import re


def non_compliant(input):
    re.match(r'(', input)  # Noncompliant {{Fix the syntax error inside this regex.}}
    #           ^
    #           ^@-1< {{Expected ')', but found the end of the regex}}
    re.match(r'x{1,2,3}|(', input)  # Noncompliant {{Fix the syntax errors inside this regex.}}
    #               ^
    #               ^@-1< {{Expected '}', but found ','}}
    #                    ^@-2< {{Expected ')', but found the end of the regex}}
    re.match(r'$[a-z^', input)  # Noncompliant {{Fix the syntax error inside this regex.}}
    #                ^
    #                ^@-1< {{Expected ']', but found the end of the regex}}
    re.match(r'(\w+-(\d+)', input)  # Noncompliant {{Fix the syntax error inside this regex.}}
    #                    ^
    #                    ^@-1< {{Expected ')', but found the end of the regex}}


def compliant(input):
    re.match(r'\(\[', input)


def unsupported_feature(input):
    re.match(r'\p{Lower}', input)  # Noncompliant
    re.match(r'(?>x)', input)  # Noncompliant
    re.match(r'x*+', input)  # Noncompliant


def false_positives():
    re.compile(r"\s*([ACGT])\s*[[]*[|]*\s*([0-9.\s]+)\s*[]]*\s*")  # Noncompliant
    # Noncompliant@+3
    re.compile(r'''
        ([A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*|\*)      # "en", "en-au", "x-y-z", "es-419", "*"
        (?:\s*;\s*q=(0(?:\.\d{,3})?|1(?:\.0{,3})?))?  # Optional "q=1.00", "q=0.8"
        (?:\s*,\s*|$)                                 # Multiple accepts per header.
        ''', re.VERBOSE)
    re.compile(r'^\s+\[([\s*[0-9]*)\] ([a-zA-Z0-9_]*)')  # Noncompliant
    re.compile(r'([^,[\]]*)(\[([^\]]+)\])?$')  # Noncompliant
    re.compile(r'{.*}')  # Noncompliant
    # Noncompliant@+3
    re.compile(
        r"""
        # Start with a literal "@{".
        @\{
          # Group at least 1 symbol, not "}".
          ([^}]+)
        # Followed by a closing "}"
        \}
        """,
        flags=re.VERBOSE)
