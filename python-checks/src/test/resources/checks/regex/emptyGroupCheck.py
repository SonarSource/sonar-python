import re


def non_compliant(input):
    re.match(r"foo()bar", input)  # Noncompliant {{Remove this empty group.}}
    #             ^^
    re.match(r"foo(?:)bar", input)  # Noncompliant
    #             ^^^^
    re.match(r"foo(?=)bar", input)  # Noncompliant
    #             ^^^^
    re.match(r"foo(?!)bar", input)  # Noncompliant
    #             ^^^^
    re.match(r"foo(?<=)bar", input)  # Noncompliant
    #             ^^^^^
    re.match(r"foo(?<!)bar", input)  # Noncompliant
    #             ^^^^^

    re.match(r"(foo()bar)", input)  # Noncompliant
    #              ^^
    re.match(r"(foo(?:)bar)", input)  # Noncompliant
    #              ^^^^
    re.match(r"(foo(?=)bar)", input)  # Noncompliant
    #              ^^^^
    re.match(r"(foo(?!)bar)", input)  # Noncompliant
    #              ^^^^
    re.match(r"(foo(?<=)bar)", input)  # Noncompliant
    #              ^^^^^
    re.match(r"(foo(?<!)bar)", input)  # Noncompliant
    #              ^^^^^


def compliant(input):
    re.match(r"foo(?-)bar", input)
    re.match(r"foo(?-x)bar", input)
    re.match(r"(foo(?-)bar)", input)

    re.match(r"foo(x)bar", input)
    re.match(r"foo(?:x)bar", input)
    re.match(r"foo(?>x)bar", input)
    re.match(r"foo(?=x)bar", input)
    re.match(r"foo(?!x)bar", input)
    re.match(r"foo(?<=x)bar", input)
    re.match(r"foo(?<!x)bar", input)

    re.match(r"[foo()bar]", input)
    re.match(r"[foo(?-)bar]", input)
    re.match(r"[foo(?:)bar]", input)
    re.match(r"[foo(?>)bar]", input)
    re.match(r"[foo(?=x)bar]", input)
    re.match(r"[foo(?!x)bar]", input)
    re.match(r"[foo(?<=x)bar]", input)
    re.match(r"[foo(?<!x)bar]", input)

    re.match(r"(foo(|)bar)", input)
    re.match(r"(foo(?-|)bar)", input)
    re.match(r"(foo(?:|)bar)", input)
    re.match(r"(foo(?>|)bar)", input)
    re.match(r"(foo(?=|)bar)", input)
    re.match(r"(foo(?!|)bar)", input)
    re.match(r"(foo(?<=|)bar)", input)
    re.match(r"(foo(?<!|)bar)", input)
