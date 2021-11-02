import re


def non_compliant(input):
   re.match(r"(?=a)b", input)  # Noncompliant {{Remove or fix this lookahead assertion that can never be true.}}
   #          ^^^^^
   re.match(r"(?=ac)ab", input)  # Noncompliant
   re.match(r"(?=a)bc", input)  # Noncompliant
   re.match(r"(?!a)a", input)  # Noncompliant
   re.match(r"(?!ab)ab", input)  # Noncompliant
   re.match(r"(?=a)[^ba]", input)  # Noncompliant
   re.match(r"(?!.)ab", input)  # Noncompliant


def compliant(input):
   re.match(r"(?=a)a", input)
   re.match(r"(?=a)..", input)
   re.match(r"(?=a)ab", input)
   re.match(r"(?!ab)..", input)
   re.match(r"(?<=a)b", input)
   re.match(r"a(?=b)", input)
   re.match(r"(?=abc)ab", input)
