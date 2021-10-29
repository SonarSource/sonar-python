import re


def non_compliant(input):
    re.match(r"a|b|c", input)  # Noncompliant {{Replace this alternation with a character class.}}
    #          ^^^^^
    re.match(r"a|(b|c)", input)  # Noncompliant
    re.match(r"abcd|(e|f)gh", input)  # Noncompliant
    re.match(r"(a|b|c)*", input)  # Noncompliant
    re.match(r"\d|x", input)  # Noncompliant
    re.match(r"\u1234|\x{12345}", input)  # Noncompliant
    re.match(r"😂|😊", input)  # Noncompliant
    re.match(r"\ud800\udc00|\udbff\udfff", input)  # Noncompliant


def compliant(input):
    re.match(r"[abc]", input)
    re.match(r"[a-c]", input)
    re.match(r"ab|cd", input)
    re.match(r"^|$", input)
    re.match(r"|", input)
