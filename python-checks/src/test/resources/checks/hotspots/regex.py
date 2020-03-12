from django.core.validators import RegexValidator
from django.urls import re_path

def build_validator(regex):
    RegexValidator(regex)  # OK

RegexValidator()
RegexValidator('')
RegexValidator('a+')
RegexValidator('a*')
RegexValidator('a{1}')
RegexValidator('(a*)*')  # Noncompliant {{Make sure that using a regular expression is safe here.}}
#              ^^^^^^^
RegexValidator('(a{1})*') # Noncompliant
RegexValidator('((a+)*') # Noncompliant
RegexValidator('(a{1})+') # Noncompliant
RegexValidator('(a+)+') # Noncompliant
RegexValidator('(a{1}){2}') # Noncompliant
RegexValidator(x:='(a+)+') # Noncompliant
RegexValidator((x:='(a+)+')) # Noncompliant
RegexValidator(x:='a+')
RegexValidator(42)


def define_http_endpoint(path, view):
    re_path(path, view)  # OK
    regexp = '(a*)*'
    RegexValidator(regexp) # Noncompliant [[secondary=-1]]
#                  ^^^^^^
    RegexValidator(*path)
    something = 42
    RegexValidator(something)

import re
from re import compile

def dynamic_pattern():
    pattern = '(a*)*'
    re.compile(pattern)  # Noncompliant
    compile(pattern) # FN - Same name as built-in symbol compile
    regexpressions.map(compile) # OK

import regex
from regex import subf


def dynamic_pattern(pattern):
    regex.subf(pattern, replacement, input)  # OK
    subf(pattern, replacement, input)  # OK


beforethisafter = r'\s*(?P<before>%s(?=\s*(\b(%s)\b)))' + \
    r'\s*(?P<this>(\b(%s)\b))' + \
    r'\s*(?P<after>%s)\s*\Z'
##
fortrantypes = r'character|logical|integer|real|complex|double\s*(precision\s*(complex|)|complex)|type(?=\s*\([\w\s,=(*)]*\))|byte'
typespattern = re.compile(
    beforethisafter % ('', fortrantypes, fortrantypes, '.*'), re.I) # Noncompliant

def test_valid_numpy_version():
    version_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+(|a[0-9]|b[0-9]|rc[0-9])"
    if np.version.release:
        res = re.match(version_pattern, np.__version__) # Noncompliant
    else:
        res = re.match(version_pattern + dev_suffix, np.__version__) # Noncompliant

def formatted_regex():
  pattern = re.compile(r'{}-\d+-{}$'.format(foo, bar))
