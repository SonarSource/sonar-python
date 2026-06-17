import re


def re_compile(_input):
    p1 = re.compile(r"\s*\s*+,")  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    p2 = re.compile(r"\s*\s*+,")  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    p3 = re.compile(r"\s*\s*+,")  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    p1.fullmatch(_input)
    p2.match(_input)
    p3.search(_input)


def full_and_partial_match(_input):
    re.search(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.findall(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.split(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.sub(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.subn(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}


def always_quadratic(_input):
    re.fullmatch(r"x*\w*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r".*.*X", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r".*.*X", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"x*a*x*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"x*(xy?)*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"x*xx*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}


def always_quadratic_other_match_types(_input):
    re.match(r"x*\w*", _input)  # OK
    re.search(r"x*\w*", _input)  # OK
    re.match(r"x*a*x*", _input)  # OK
    re.search(r"x*a*x*", _input)  # OK
    re.match(r"x*(xy?)*", _input)  # OK
    re.search(r"x*(xy?)*", _input)  # OK
    re.match(r"x*xx*", _input)  # OK
    # FP
    re.search(r"x*xx*", _input)  # Noncompliant


def non_possessive_followed_by_possessive(_input):
    re.fullmatch(r".*\s*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r".*\s*", _input)  # OK
    re.search(r".*\s*", _input)  # OK
    re.fullmatch(r".*\s*+", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r".*\s*+", _input)  # OK
    re.search(r".*\s*+", _input)  # OK
    re.fullmatch(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.search(r"\s*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"[a\s]*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"[a\s]*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.search(r"[a\s]*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"[a\s]*b*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"[a\s]*b*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.search(r"[a\s]*b*\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}


def implicit_reluctant_quantifier_in_partial_match(_input):
    re.search(r"\s*,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"\s*,", _input)  # OK
    re.fullmatch(r"\s*,", _input)  # OK
    re.search(r"\s*+,", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"\s*+,", _input)  # OK
    re.fullmatch(r"\s*+,", _input)  # OK
    re.search(r"(?s:.*)\s*,(?s:.*)", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"(?s:.*)\s*,(?s:.*)", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"(?s:.*)\s*,(?s:.*)", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.search(r"(?s:.*)\s*+,(?s:.*)", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"(?s:.*)\s*+,(?s:.*)", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"(?s:.*)\s*+,(?s:.*)", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}


def different_polynomials(_input):
    re.fullmatch(r"x*x*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"x*x*", _input)  # OK
    re.search(r"x*x*", _input)  # OK
    re.fullmatch(r"x*x*x*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"x*x*x*", _input)  # OK
    re.search(r"x*x*x*", _input)  # OK
    re.fullmatch(r"x*x*x*x*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"x*x*x*x*", _input)  # OK
    re.search(r"x*x*x*x*", _input)  # OK
    re.fullmatch(r"x*x*x*x*x*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"x*x*x*x*x*", _input)  # OK
    re.search(r"x*x*x*x*x*", _input)  # OK
    re.fullmatch(r"[^=]*.*.*=.*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.match(r"[^=]*.*.*=.*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.search(r"[^=]*.*.*=.*", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.search(r"a+b", _input) # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}


def linear_when_optimized_reported_by_s5852(_input):
    re.search(r"(.?,)*X", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.search(r"(.?,)*\1", _input)  # Noncompliant  {{Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.}}
    re.fullmatch(r"(.?,)*X", _input)  # OK, LINEAR_WHEN_OPTIMIZED reported by S5852
    re.match(r"(.?,)*X", _input)  # OK
    re.fullmatch(r"(.?,)*\1", _input)  # OK
    re.match(r"(.?,)*\1", _input)  # OK


def should_not_report_exponential_cases(_input):
    re.fullmatch(r"(.*,)*?", _input)  # OK, exponential cases reported by S5852
    re.fullmatch(r"(.*,)*", _input)  # OK
    re.fullmatch(r"(.*,)*X", _input)  # OK
    re.match(r"(.*,)*X", _input)  # OK
    re.search(r"(.*,)*X", _input)  # OK


def compliant(_input):
    re.fullmatch(r"abc", _input)
    re.fullmatch(r"(?s)(.*,)*.*", _input)
    re.fullmatch(r"(?:(.)\1)*", _input)
