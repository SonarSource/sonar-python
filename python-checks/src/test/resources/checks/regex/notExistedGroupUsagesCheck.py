import re

def pattern_compile_group_by_name():
    pattern = re.compile(r"(?P<a>.)")
    #                      ^^^^^^^^> 1 {{There is no group named 'b' in the regular expression.}}
    matches = pattern.match("abc")
    g = matches.group("b") # Noncompliant {{There is no group named 'b' in the regular expression.}}
    #                 ^^^ 1

def pattern_compile_group_by_number():
    pattern = re.compile(r"(?P<a>.)")
    #                      ^^^^^^^^> 1 {{There is no group with number '2' in the regular expression.}}
    matches = pattern.match("abc")
    g = matches.group(2) # Noncompliant {{There is no group with number '2' in the regular expression.}}
    #                 ^ 1

def match_group_by_name():
    matches2 = re.match(r"(?P<a>.)", "abc")
    #                     ^^^^^^^^> 1 {{There is no group named 'b' in the regular expression.}}
    g = matches2.group("b") # Noncompliant {{There is no group named 'b' in the regular expression.}}
    #                  ^^^ 1

def match_group_by_number():
    matches2 = re.match(r"(?P<a>.)", "abc")
    #                     ^^^^^^^^> 1 {{There is no group with number '2' in the regular expression.}}
    g = matches2.group(2) # Noncompliant {{There is no group with number '2' in the regular expression.}}
    #                  ^ 1

def compliant(a):
    pattern = re.compile(r"(?P<a>.)")
    matches = pattern.match("abc")
    g1 = matches.group("a")
    g2 = matches.group(1)

    matches2 = re.match(r"(?P<a>.)", "abc")
    g3 = matches2.group("a")
    g4 = matches2.group(1)

def multiple_pattern_assignments(a):
    if a:
        pattern = re.compile(r"(?P<a>.)")
    else:
        pattern = re.compile(r"(?P<b>.)")
    matches = pattern.match("abc")
    g1 = matches.group("b")
    g2 = matches.group(2)

    if a:
        matches2 = re.match(r"(?P<a>.)", "abc")
    else:
        matches2 = re.match(r"(?P<b>.)", "abc")
    g3 = matches2.group("a")
    g4 = matches2.group(1)

def multiple_match_assignments(a):
    matches = re.match(r"(?P<a>.)", "abc")
    g1 = matches.group("a")
    pattern = re.compile(r"(?P<b>.)")
    matches = pattern.match("abc")
    g1 = matches.group("a") # FN

def group_name_never_user(a):
    matches = re.match(r"(?P<a>.)(?P<b>.)", "abc") # FN https://sonarsource.atlassian.net/browse/SONARPY-1322
    g1 = matches.group("b")
