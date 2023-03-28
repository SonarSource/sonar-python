import re

def pattern_compile_group_by_name():
    pattern = re.compile(r"(?P<a>.)bc")
    #                      ^^^^^^^^> 1 {{Named group 'a'}}
    matches = pattern.match("abc")
    g = matches.group("b") # Noncompliant {{There is no group named 'b' in the regular expression.}}
    #                 ^^^ 1

def pattern_compile_group_by_number():
    pattern = re.compile(r"(?P<a>.)bc")
    #                      ^^^^^^^^> 1 {{Group 1}}
    matches = pattern.match("abc")
    g = matches.group(1) # Noncompliant {{Directly use 'a' instead of its group number.}}
    #                 ^ 1

def pattern_compile_group_by_name_no_named_groups():
    pattern = re.compile(r"(.)bc")
    #                      ^^^^^> 1 {{No named groups defined in this regular expression.}}
    matches = pattern.match("abc")
    g = matches.group("b") # Noncompliant {{There is no group named 'b' in the regular expression.}}
    #                 ^^^ 1

def match_group_by_name():
    matches2 = re.match(r"(?P<a>.)bc", "abc")
    #                     ^^^^^^^^> 1 {{Named group 'a'}}
    g = matches2.group("b") # Noncompliant {{There is no group named 'b' in the regular expression.}}
    #                  ^^^ 1

def match_group_by_number():
    matches2 = re.match(r"(?P<a>.)bc", "abc")
    #                     ^^^^^^^^> 1 {{Group 1}}
    g = matches2.group(1) # Noncompliant {{Directly use 'a' instead of its group number.}}
    #                  ^ 1

def match_group_by_name_no_named_groups():
    matches2 = re.match(r"(.)bc", "abc")
    #                     ^^^^^> 1 {{No named groups defined in this regular expression.}}
    g = matches2.group("b") # Noncompliant {{There is no group named 'b' in the regular expression.}}
    #                  ^^^ 1

def compliant(a):
    pattern = re.compile(r"(?P<a>.)")
    matches = pattern.match("abc")
    g1 = matches.group("a")

    matches2 = re.match(r"(?P<a>.)", "abc")
    g3 = matches2.group("a")

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
    matches = re.match(r"(?P<a>.)(?P<b>.)", "abc") # FN SONARPY-1322
    g1 = matches.group("b")
