import re

def non_compliant(input):
    re.match(r"xx*", input) # Noncompliant {{Use simple repetition 'x+' instead of 'xx*'.}}
    #          ^
    re.match(r"[ah-hz]", input) # Noncompliant {{Use simple character 'h' instead of 'h-h'.}}
    #            ^^^
    re.match(r"[\s\S]", input, re.DOTALL)  # Noncompliant {{Use concise character class syntax '.' instead of '[\s\S]'.}}
    #          ^^^^^^
    re.match(r"[\d\D]", input)  # Noncompliant {{Use concise character class syntax '.' instead of '[\d\D]'.}}
    re.match(r"[\w\W]", input)  # Noncompliant {{Use concise character class syntax '.' instead of '[\w\W]'.}}
    re.match(r"[0-9]", input)  # Noncompliant {{Use concise character class syntax '\d' instead of '[0-9]'.}}
    re.match(r"foo[0-9]barr", input)  # Noncompliant
    #             ^^^^^
    re.match(r"[^0-9]", input)  # Noncompliant {{Use concise character class syntax '\D' instead of '[^0-9]'.}}
    re.match(r"[A-Za-z0-9_]",input)  # Noncompliant {{Use concise character class syntax '\w' instead of '[A-Za-z0-9_]'.}}
    re.match(r"[0-9_A-Za-z]", input)  # Noncompliant
    re.match(r"[^A-Za-z0-9_]", input)  # Noncompliant {{Use concise character class syntax '\W' instead of '[^A-Za-z0-9_]'.}}
    re.match(r"[^0-9_A-Za-z]", input)  # Noncompliant
    re.match(r"x{0,1}", input)  # Noncompliant {{Use concise quantifier syntax '?' instead of '{0,1}'.}}
    re.match(r"x{0,1}?", input)  # Noncompliant
    re.match(r"x{0,}", input)  # Noncompliant {{Use concise quantifier syntax '*' instead of '{0,}'.}}
    re.match(r"x{0,}?", input)  # Noncompliant
    re.match(r"x{1,}", input)  # Noncompliant {{Use concise quantifier syntax '+' instead of '{1,}'.}}
    re.match(r"x{1,}?", input)  # Noncompliant
    re.match(r"x{2,2}", input)  # Noncompliant {{Use concise quantifier syntax '{2}' instead of '{2,2}'.}}
    re.match(r"x{2,2}?", input)  # Noncompliant


def compliant(input):
    re.match(r"[x]", input)
    re.match(r"[12]", input)
    re.match(r"[1234]", input)
    re.match(r"[1-3]", input)
    re.match(r"[1-9abc]", input)
    re.match(r"[1-9a-bAB]", input)
    re.match(r"[1-9a-bA-Z!]", input)
    re.match(r"[1-2[a][b][c]]", input)
    re.match(r"[0-9[a][b][c]]", input)
    re.match(r"[0-9a-z[b][c]]", input)
    re.match(r"[0-9a-zA-Z[c]]", input)
    re.match(r"x?", input)
    re.match(r"x*", input)
    re.match(r"x+", input)
    re.match(r"x{2}", input)
    re.match(r"[\s\S]", input)
    re.match(r"[\w\S]", input)
    re.match(r"[\d\S]", input)
    re.match(r"[\s\d]", input)
    re.match(r"xx+", input)
    re.match(r"xy*", input)
