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


def unsupported_feature(input):
    re.match(r'\p{Lower}', input)  # Noncompliant
    re.match(r'(?>x)', input)  # Noncompliant
    re.match(r'x*+', input)  # Noncompliant
