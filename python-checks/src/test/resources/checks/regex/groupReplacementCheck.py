import re


def non_compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"

    re.sub(r"(a)(b)(c)", r"\1, \9, \3", "abc")  # Noncompliant
    #                    ^^^^^^^^^^^^^
    re.sub("(a)", r"\2", "")  # Noncompliant {{Referencing non-existing group: 2.}}
    #             ^^^^^
    re.sub("(a)", r'${2}', "")  # Noncompliant
    re.sub("a", r"$1", "")  # Noncompliant
    re.sub("(?!a)", r"$1", "")  # Noncompliant
    re.sub("(a)", r'\2', "")  # Noncompliant
    re.sub("(a)", r"$1 $2", "")  # Noncompliant
    re.sub("(a)", r"$3 $2", "")  # Noncompliant {{Referencing non-existing groups: 3, 2.}}
    re.sub("(a)", r"$2 \1", "")  # Noncompliant


def compliant():
    input = "Bob is a Bird... Bob is a Plane... Bob is Superman!"
    re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)

    re.sub("(a)", r"$0", "")
    re.sub("(a)", r"$1", "")
    re.sub("(a)", r'${1}', "")
    re.sub("(a)", r"\1", "")
    re.sub("(a)(b)", r"\1 \2", "")
    re.sub("(a(b))", r"\1 \2", "")
