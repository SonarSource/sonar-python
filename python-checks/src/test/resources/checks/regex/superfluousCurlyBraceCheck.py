import re


def non_compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"

    changed = re.match(r"(abc){1}", input)  # Noncompliant {{Remove this unnecessary quantifier.}}
    #                         ^^^
    changed = re.match(r"(abc){1,1}", input)  # Noncompliant {{Remove this unnecessary quantifier.}}
    #                         ^^^^^
    changed = re.match(r"(abc){0}", input)  # Noncompliant {{Remove this unnecessarily quantified expression.}}
    #                    ^^^^^^^^
    changed = re.match(r"(abc){0,0}", input)  # Noncompliant {{Remove this unnecessarily quantified expression.}}
    #                    ^^^^^^^^^^


def compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman! abcabcabcabcabc"
    changed = re.match(r"(abc){0,}", input)
    changed = re.match(r"(abc){1,}", input)
    changed = re.match(r"(abc){1,2,3}", input)

    changed = re.match(r"(abc){0,}", input)
    changed = re.match(r"(abc){2}", input)
    changed = re.match(r"(abc){1,2}", input)
    changed = re.match(r"(abc){0,1}", input)


def coverage(input):
    re.match(r"Bob is", input)
