import re

re.match(r'.*', "foo", re.I); # Noncompliant {{CASE_INSENSITIVE|UNICODE_CASE|UNICODE_CHARACTER_CLASS}}
re.match(r'.*', "foo", re.IGNORECASE); # Noncompliant {{CASE_INSENSITIVE|UNICODE_CASE|UNICODE_CHARACTER_CLASS}}

re.match(r'.*', "foo", re.M); # Noncompliant {{MULTILINE|UNICODE_CASE|UNICODE_CHARACTER_CLASS}}
re.match(r'.*', "foo", re.MULTILINE); # Noncompliant {{MULTILINE|UNICODE_CASE|UNICODE_CHARACTER_CLASS}}

re.match(r'.*', "foo", re.S); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS|DOTALL}}
re.match(r'.*', "foo", re.DOTALL); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS|DOTALL}}

re.match(r'.*', "foo", re.X); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS|VERBOSE}}
re.match(r'.*', "foo", re.VERBOSE); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS|VERBOSE}}

re.match(r'.*', "foo", re.U); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS}}
re.match(r'.*', "foo", re.UNICODE); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS}}

re.match(r'.*', "foo", re.A); # Noncompliant {{ASCII}}
re.match(r'.*', "foo", re.ASCII); # Noncompliant {{ASCII}}

re.match(r'.*', "foo", re.UNKNOWN); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS}}
re.match(r'.*', "foo", not_re.UNKNOWN); # Noncompliant {{UNICODE_CASE|UNICODE_CHARACTER_CLASS}}
re.match(r'.*', "foo", re.I|UNKNOWN); # Noncompliant {{CASE_INSENSITIVE|UNICODE_CASE|UNICODE_CHARACTER_CLASS}}

re.match(r'.*', "foo", re.I|re.M); # Noncompliant {{CASE_INSENSITIVE|MULTILINE|UNICODE_CASE|UNICODE_CHARACTER_CLASS}}