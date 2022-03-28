import re


def non_compliant():
    input = "Hello world!  Bob is a Bird... Bob is a Plane... Bob is Superman!"
    changed = re.match(r"Hello,   world!", input)  # Noncompliant {{Replace spaces with quantifier `{3}`.}}
    #                           ^^
    changed = re.match(r"Hello,  world!", input)  # Noncompliant {{Replace spaces with quantifier `{2}`.}}
    #                           ^
    changed = re.match(r"Hello, world!     ", input)  # Noncompliant {{Replace spaces with quantifier `{5}`.}}
    #                                  ^^^^
    changed = re.match(r"Hello,      world!     ", input)  # Noncompliant 2


def compliant():
    input = "Hello world!  Bob is a Bird... Bob is a Plane... Bob is Superman!"
    changed = re.match(r"\r\n|\r", input, re.MS)
    changed = re.match(r"[   ]", input)
    changed = re.compile(r"[   ]", re.X)
    changed = re.match(r"Hello, world!\t\t\t\t", input)
    changed = re.match(r"Hello , world!", input)
    changed = re.match(r"'Hello, {3}world!'", input)
    changed = re.match(r'Hello,     world!', flags=re.VERBOSE)
    changed = re.compile(r'Hello,    world!', flags=re.X)
