import re


def non_compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"

    re.sub(r"(a)(b)(c)", r"\1, \9, \3", "abc")  # Noncompliant
    #                    ^^^^^^^^^^^^^
    re.sub("(a)", r"\2", input)  # Noncompliant {{Referencing non-existing group: 2.}}
    #             ^^^^^
    re.sub("(a)", r'\g<2>', input)  # Noncompliant
    re.sub("a", r"\g<1>", input)  # Noncompliant
    re.sub("(?!a)", r"\g<1>", input)  # Noncompliant
    re.sub("(a)", r'\2', input)  # Noncompliant
    re.sub("(a)", r"\g<2> \2", input)  # Noncompliant
    re.sub("(a)", r"\3 \g<2>", input)  # Noncompliant {{Referencing non-existing groups: 3, 2.}}
    re.sub("(a)", r"\2 \1", input)  # Noncompliant
    re.sub("(s)", r"\12 \1", input)  # Noncompliant


def compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"
    re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)

    re.sub("(a)", r"\0c", input)
    re.sub("(a)", r"\g<0>c", input)
    re.sub("(a)", r'\1', input)
    re.sub("(a)", r"\g<1>ttt", input)
    re.sub("(a)(b)", r"\g<1>aaa \2bbb", input)
    re.sub("(a(n))", r"\1a \g<2>", input)

    # coverage
    re.match("up", input)
    pattern = r"a"
    re.sub("(ab)", pattern, input)
    re.sub("(a)", r"[\?]", input)
    re.sub("(A)")
