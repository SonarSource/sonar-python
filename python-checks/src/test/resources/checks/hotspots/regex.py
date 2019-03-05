from django.core.validators import RegexValidator
from django.urls import re_path

def build_validator(regex):
    RegexValidator(regex)  # Noncompliant {{Make sure that using a regular expression is safe here.}}
#   ^^^^^^^^^^^^^^^^^^^^^

RegexValidator('(a*)*')  # Noncompliant

def define_http_endpoint(path, view):
    re_path(path, view)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^

import re
from re import compile

def dynamic_pattern(pattern):
    re.compile(pattern)  # Noncompliant
    compile(pattern) # Noncompliant
    regexpressions.map(compile) # Noncompliant
#                      ^^^^^^^

import regex
from regex import subf


def dynamic_pattern(pattern):
    regex.subf(pattern, replacement, input)  # Noncompliant
    subf(pattern, replacement, input)  # Noncompliant
