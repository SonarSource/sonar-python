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
    # atomic group
    re.match(r'(?>x)', input)
    # possessive quantifier
    re.match(r'x*+', input)

def false_positives():
    re.compile(r'''
          # Match tail of: [text][id]
          [ ]?          # one optional space
          (?:\n[ ]*)?   # one optional newline followed by spaces
          \[
            (?P<id>.*?)
          \]
        ''', re.X | re.S)
    # Noncompliant@-5
