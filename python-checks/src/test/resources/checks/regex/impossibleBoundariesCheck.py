import re


def non_compliant(input):
    re.match(r'$[a-z]^', input)  # Noncompliant 2
    #          ^0
    re.match(r'$[a-z]', input)  # Noncompliant {{Remove or replace this boundary that will never match because it appears before mandatory input.}}
    re.match(r'$(abc)', input)  # Noncompliant
    re.match(r'[a-z]^', input)  # Noncompliant
    re.match(r'\Z[a-z]', input)  # Noncompliant
    re.match(r'\z[a-z]', input)  # Noncompliant
    re.match(r'[a-z]\A', input)  # Noncompliant
    re.match(r'($)a', input)  # Noncompliant
    re.match(r'a$|$a', input)  # Noncompliant
    re.match(r'^a|a^', input)  # Noncompliant
    re.match(r'a(b|^)', input)  # Noncompliant
    re.match(r'(?=abc^)', input)  # Noncompliant
    re.match(r'(?!abc^)', input)  # Noncompliant
    re.match(r'abc(?=^abc)', input)  # Noncompliant
    re.match(r'abc(?<=$abc)', input)  # Noncompliant
    re.match(r'abc(?<=abc$)def', input)  # Noncompliant
    re.match(r'(?:abc(X|^))*Y?', input)  # Noncompliant

    re.match(r'a\Z\nb', input, re.MULTILINE)  # Noncompliant
    re.match(r'a\zb', input, re.MULTILINE)  # Noncompliant
    re.match(r'a\n\Ab', input, re.MULTILINE)  # Noncompliant

    # False positives because the end delimiter does not capture the newlines (SONARPHP-1238)
    re.match(r'a$./s', input)  # Noncompliant


def probably_non_compliant(input):
    re.match(r'$.*', input)  # Noncompliant {{Remove or replace this boundary that can only match if the previous part matched the empty string because it appears before mandatory input.}}
    re.match(r'$.?', input)  # Noncompliant

    re.match(r'$a*', input)  # Noncompliant
    re.match(r'$a?', input)  # Noncompliant
    re.match(r'$[abc]*', input)  # Noncompliant
    re.match(r'$[abc]?', input)  # Noncompliant

    re.match(r'.*^', input)  # Noncompliant {{Remove or replace this boundary that can only match if the previous part matched the empty string because it appears after mandatory input.}}
    re.match(r'.?^', input)  # Noncompliant

    re.match(r'a*^', input)  # Noncompliant
    re.match(r'a?^', input)  # Noncompliant
    re.match(r'[abc]*^', input)  # Noncompliant
    re.match(r'[abc]?^', input)  # Noncompliant

    re.match(r'$.*^', input)  # Noncompliant 2
    re.match(r'$.?^', input)  # Noncompliant 2
    re.match(r'$a*^', input)  # Noncompliant 2
    re.match(r'$a?^', input)  # Noncompliant 2
    re.match(r'$[abc]*^', input)  # Noncompliant 2
    re.match(r'$[abc]?^', input)  # Noncompliant 2


def compliant(input):
    re.match(r'^[a-z]$', input)
    re.match(r'^$', input)
    re.match(r'^(?i)$', input)
    re.match(r'^$(?i)', input)
    re.match(r'^abc$|^def$', input)
    re.match(r'(?i)^abc$', input)
    re.match(r'()^abc$', input)
    re.match(r'^abc$()', input)
    re.match(r'^abc$\b', input)
    re.match(r'(?=abc)^abc$', input)
    re.match(r'(?=^abc$)abc', input)
    re.match(r'(?!^abc$)abc', input)
    re.match(r'abc(?<=^abc$)', input)
    re.match(r'^\d$(?<!3)', input)
    re.match(r'(?=$)', input)
    re.match(r"(?i)(true)(?=(?:[^']|'[^']*')*$)", input)
    re.match(r'(?:abc(X|$))*Y?', input)
    re.match(r'(?:x*(Xab|^)abc)*Y?', input)
    re.match(r'a$\nb', input, re.MULTILINE)
    re.match(r'a\n^b', input, re.MULTILINE)
    re.compile(r"(\d+)(\s+.*)$                         # score, vulgar components", re.VERBOSE)
