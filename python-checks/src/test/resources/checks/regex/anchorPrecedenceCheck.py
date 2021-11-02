import re


def noncompliant(input):
    re.match(r"^a|b|c$", input)  # Noncompliant {{Group parts of the regex together to make the intended operator precedence explicit.}}
    #          ^^^^^^^
    re.match(r"^a|b|cd", input)  # Noncompliant
    re.match(r"(?i)^a|b|cd", input)  # Noncompliant
    re.match(r"(?i:^a|b|cd)", input)  # Noncompliant
    re.match(r"a|b|c$", input)  # Noncompliant
    re.match(r"\Aa|b|c\Z", input)  # Noncompliant
    re.match(r"\Aa|b|c\z", input)  # Noncompliant


def compliant(input):
    re.match(r"^(?:a|b|c)$", input)
    re.match(r"(?:^a)|b|(?:c$)", input)
    re.match(r"^abc$", input)
    re.match(r"a|b|c", input)
    re.match(r"^a$|^b$|^c$", input)
    re.match(r"^a$|b|c", input)
    re.match(r"a|b|^c$", input)
    re.match(r"^a|^b$|c$", input)
    re.match(r"^a|^b|c$", input)
    re.match(r"^a|b$|c$", input)
    # Only beginning and end of line/input boundaries are considered - not word boundaries
    re.match(r"\ba|b|c\b", input)
    re.match(r"\ba\b|\bb\b|\bc\b", input)
    # If multiple alternatives are anchored, but not all, that's more likely to be intentional than if only the first
    # one were anchored, so we won't report an issue for the following line:
    re.match(r"^a|^b|c", input)
    re.match(r"aa|bb|cc", input)
    re.match(r"^", input)
    re.match(r"^[abc]$", input)
    re.match(r"|", input)
    re.match(r"[", input)
    re.match(r"(?i:^)a|b|c", input)  # False negative; we don't find the anchor if it's hidden inside a sub-expression
