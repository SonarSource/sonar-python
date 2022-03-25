import re


def non_compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"
    changed = re.match(r"[0]", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                     ^

    changed = re.match(r"[B]", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                     ^
    changed = re.match(r"[ ]", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                     ^
    changed = re.match(r"[:]", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                     ^
    changed = re.match(r"([)])", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                      ^
    changed = re.match(r"[:]", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                     ^
    changed = re.match(r"[]]", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                     ^
    changed = re.match(r"([\[])", input)  # Noncompliant {{Replace this character class by the character itself.}}
    #                      ^^
    changed = re.match(r'[b][c]', input, re.M) # Noncompliant 2


def compliant():
    input = "abcdefghijklmnopqa"
    changed = re.match(re.escape("test"), input)
    changed = re.match(r"[abc]", input)
    changed = re.match(r"[a-c]", input)
    changed = re.match(r"ab|cd", input)
    changed = re.match(r"^|$", input)
    changed = re.match(r"|", input)
    changed = re.match(r"[\[a\]]", input)
    # # Special characters do not raise warning
    changed = re.match(r"a[.]a", input)
    changed = re.match(r"a[*]a", input)
    changed = re.match(r"a[+]a", input)
    changed = re.match(r"a[^]a", input)
    changed = re.match(r"a[{m}]a", input)
    changed = re.match(r"a[\d]a", input)
    changed = re.match(r"a[\w]a", input)
    changed = re.match(r"a[?]a", input)
    changed = re.match(r"a[|]a", input)
    changed = re.match(r"a[\W]a", input)
    changed = re.match(r"a[\]a", input)
    changed = re.match(r"a a", input)

    changed = re.compile(r'[ \t # comment]', re.X)

    changed = re.compile(r'[ \t  ]', re.X)
    changed = re.compile(r'[ \t]', re.X)
    changed = re.match(r'[ \t]', input, re.X)

    # TODO : False Negatives. We deactivated the SingleCharCharacterClassFinder whenever the flag X or VERBOSE is set.
    changed = re.compile(r'[\t]', re.X)
    changed = re.compile(r'[a]', re.VERBOSE)
    changed = re.match(r'[b][c]', input, re.M | re.X)
