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



def define_http_endpoint(path, view):
    re_path(path, view)  # OK
    regexp = '(a*)*'
    RegexValidator(regexp) # Noncompliant [[secondary=-1]]
#                  ^^^^^^

import re
from re import compile

def dynamic_pattern():
    pattern = '(a*)*'
    re.compile(pattern)  # Noncompliant
    compile(pattern) # Noncompliant
    regexpressions.map(compile) # OK

import regex
from regex import subf


def dynamic_pattern(pattern):
    regex.subf(pattern, replacement, input)  # OK
    subf(pattern, replacement, input)  # OK
