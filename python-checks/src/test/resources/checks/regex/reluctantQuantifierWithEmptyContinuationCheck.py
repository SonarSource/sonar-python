import re


def non_compliant(input):
    re.match(r".*?x?", input)  # Noncompliant {{Fix this reluctant quantifier that will only ever match 0 repetitions.}}
    #          ^^^
    re.match(r".+?x?", input)  # Noncompliant {{Fix this reluctant quantifier that will only ever match 1 repetition.}}
    re.match(r".{2,4}?x?", input)  # Noncompliant {{Fix this reluctant quantifier that will only ever match 2 repetitions.}}
    re.match(r".*?$", input)  # Noncompliant {{Remove the '?' from this unnecessarily reluctant quantifier.}}
    re.match(r".*?()$", input)  # Noncompliant {{Remove the '?' from this unnecessarily reluctant quantifier.}}


def compliant(input):
    re.match(r".*?x", input)
    re.match(r".*?x$", input)
    re.match(r".*?[abc]", input)
