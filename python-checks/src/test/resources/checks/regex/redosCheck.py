import re


def re_compile(_input):
    p1 = re.compile(r"(.*,)*")  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    p2 = re.compile(r"(.*,)*")
    p3 = re.compile(r"(.*,)*")
    p4 = re.compile(r"(.*,)*")  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    p5 = re.compile(r"(.*,)*")
    p6.p7 = re.compile(r"(.*,)*")
    re.compile(r"(.*,)*")  # No issue: unused
    p1.fullmatch(_input)
    p2.match(_input)
    p3.search(_input)
    p4.search(_input)
    p4.fullmatch(_input)
    unknown(p1)


def regex_in_variable(_input):
    my_regex1 = r"(.*,)*"  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
#                 ^^^^^^
    my_regex2 = r"(.*,)*"  # OK: not used in sensitive context
    p1 = re.compile(my_regex1)
    p1.fullmatch(_input)
    p2 = re.compile(my_regex2)
    p2.match(my_regex)


def full_and_partial_match(_input):
    re.fullmatch(r"(.*,)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
#                  ^^^^^^
    re.match(r"(.*,)*", _input)  # OK
    re.search(r"(.*,)*", _input)  # OK, partial match
    re.findall(r"(.*,)*", _input)  # OK, partial match
    re.split(r"(.*,)*", _input)  # OK, partial match
    re.sub(r"(.*,)*", _input)  # OK, partial match
    re.subn(r"(.*,)*", _input)  # OK, partial match
    re.search(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.findall(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.split(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.sub(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.subn(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}

def always_exponential(_input):
    re.fullmatch(r"(.*,)*?", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(.?,)*?", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(a|.a)*?", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(?:.*,)*(X)\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(?:.*,)*(X)\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(?:.*,)*(X)\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(.*,)*\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)*\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(.*,)*\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}


def always_quadratic(_input):
    re.fullmatch(r"x*\w*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r".*.*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r".*.*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"x*a*x*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"x*(xy?)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"x*xx*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}


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
    re.fullmatch(r".*\s*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r".*\s*", _input)  # OK
    re.search(r".*\s*", _input)  # OK
    re.fullmatch(r".*\s*+", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r".*\s*+", _input)  # OK
    re.search(r".*\s*+", _input)  # OK
    re.fullmatch(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"\s*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"[a\s]*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"[a\s]*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"[a\s]*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"[a\s]*b*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"[a\s]*b*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"[a\s]*b*\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}


def implicit_reluctant_quantifier_in_partial_match(_input):
    re.search(r"\s*,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"\s*,", _input)  # OK
    re.fullmatch(r"\s*,", _input)  # OK
    re.search(r"\s*+,", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"\s*+,", _input)  # OK
    re.fullmatch(r"\s*+,", _input)  # OK
    re.search(r"(?s:.*)\s*,(?s:.*)", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(?s:.*)\s*,(?s:.*)", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(?s:.*)\s*,(?s:.*)", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(?s:.*)\s*+,(?s:.*)", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(?s:.*)\s*+,(?s:.*)", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(?s:.*)\s*+,(?s:.*)", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}


def different_polynomials(_input):
    re.fullmatch(r"x*x*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"x*x*", _input)  # OK
    re.search(r"x*x*", _input)  # OK
    re.fullmatch(r"x*x*x*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"x*x*x*", _input)  # OK
    re.search(r"x*x*x*", _input)  # OK
    re.fullmatch(r"x*x*x*x*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"x*x*x*x*", _input)  # OK
    re.search(r"x*x*x*x*", _input)  # OK
    re.fullmatch(r"x*x*x*x*x*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"x*x*x*x*x*", _input)  # OK
    re.search(r"x*x*x*x*x*", _input)  # OK
    re.fullmatch(r"[^=]*.*.*=.*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"[^=]*.*.*=.*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"[^=]*.*.*=.*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}


def should_be_ok(_input):
    re.match(r"(.*,)*?", _input)
    re.match(r"(.?,)*?", _input)
    re.match(r"(a|.a)*?", _input)
    re.match(r"x*a*x*", _input)
    re.match(r"x*(xy?)*", _input)
    re.match(r"x*xx*", _input)


def java9_optimized_cases_not_optimized_in_python(_input):
    re.fullmatch(r"(.?,)*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.?,)*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(.?,)*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(.?,)*\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.?,)*\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(.?,)*\1", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to polynomial runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(?:(.?)\1,)*", _input)  # FN
    re.match(r"(?:(.?)\1,)*", _input)  # OK
    re.search(r"(?:(.?)\1,)*", _input)  # OK


def polynomial_in_java9_exponential_in_python(_input):
    re.fullmatch(r"(.*,)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)*", _input)  # OK
    re.search(r"(.*,)*", _input)  # OK
    re.fullmatch(r"(.*,)*.*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)*.*", _input)  # OK
    re.search(r"(.*,)*.*", _input)  # OK
    re.fullmatch(r"(.*,)*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(.*,)*X", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(.*?,)+", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*?,)+", _input)  # OK
    re.search(r"(.*?,)+", _input)  # OK
    re.fullmatch(r"(.*?,){5,}", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*?,){5,}", _input)  # OK
    re.search(r"(.*?,){5,}", _input)  # OK
    re.fullmatch(r"((.*,)*)?", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"((.*,)*)?", _input)  # OK
    re.search(r"((.*,)*)?", _input)  # OK
    re.fullmatch(r"(.*,)* (.*,)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)* (.*,)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(.*,)* (.*,)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(.*,)*$", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)*$", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.search(r"(.*,)*$", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.fullmatch(r"(.*,)*(..)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)*(..)*", _input)  # OK
    re.search(r"(.*,)*(..)*", _input)  # OK
    re.fullmatch(r"(.*,)*(.{2})*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(.*,)*(.{2})*", _input)  # OK
    re.search(r"(.*,)*(.{2})*", _input)  # OK


def false_positives():
    re.fullmatch(r"((.*,)*)*+", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"((.*,)*)*+", _input)  # OK
    re.search(r"((.*,)*)*+", _input)  # OK

    re.fullmatch(r"(?>(.*,)*)", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"(?>(.*,)*)", _input)  # OK
    re.search(r"(?>(.*,)*)", _input)  # OK

    re.fullmatch(r"((?>.*,)*)*", _input)  # Noncompliant  {{Make sure the regex used here, which is vulnerable to exponential runtime due to backtracking, cannot lead to denial of service.}}
    re.match(r"((?>.*,)*)*", _input)  # OK
    re.search(r"((?>.*,)*)*", _input)  # OK

def compliant(_input):
    re.fullmatch(r"abc", _input)
    re.fullmatch(r"(?s)(.*,)*.*", _input)
    re.fullmatch(r"(?:(.)\1)*", _input)
    re.fullmatch(r"(?:\1|x(.))*", _input)
    re.fullmatch(r"(?:\1|x(.))+", _input)
    re.fullmatch(r"(?:\1|x(.)){1,2}", _input)
    re.fullmatch(r"(.)(?:\1\2|x(.))*", _input)
