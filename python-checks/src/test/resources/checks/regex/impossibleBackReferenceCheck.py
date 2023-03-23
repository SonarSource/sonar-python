import re

def noncompliant():
    pattern = re.compile(r"\1") # Noncompliant {{Fix this backreference - it refers to a capturing group that doesn't exist.}}
    pattern = re.compile(r"\1(.)") # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}
    pattern = re.compile(r"(.)\2") # Noncompliant {{Fix this backreference - it refers to a capturing group that doesn't exist.}}
    pattern = re.compile(r"(.)|\1") # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}
    pattern = re.compile(r"(?P<x>.)|(?P=x)") # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}
    pattern = re.compile(r"(?P<x>.)(?P=y)") # Noncompliant {{Fix this backreference - it refers to a capturing group that doesn't exist.}}
    #                              ^^^^^^
    pattern = re.compile(r"(?P<x>.)(?P=y)(?P<y>.)") # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}
    #                              ^^^^^^
    pattern = re.compile(r"(.)\2(.)") # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}

def compliant():
    pattern = re.compile(r"(.)\1")
    pattern = re.compile(r"(?P<y>.)(?P=y)")

