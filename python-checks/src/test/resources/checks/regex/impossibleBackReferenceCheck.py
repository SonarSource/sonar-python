import re

pattern = re.compile(r"\1(.)", re.DOTALL) # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}
pattern = re.compile(r"(.)\2", re.DOTALL) # Noncompliant {{Fix this backreference - it refers to a capturing group that doesn't exist.}}
pattern = re.compile(r"(.)|\1", re.DOTALL) # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}
pattern = re.compile(r"(?P<x>.)|(?P=x)", re.DOTALL) # Noncompliant {{Fix this backreference, so that it refers to a group that can be matched before it.}}

pattern = re.compile(r"(.)\1", re.DOTALL)
pattern = re.compile(r"(?P<x>.)(?P=x)", re.DOTALL)
