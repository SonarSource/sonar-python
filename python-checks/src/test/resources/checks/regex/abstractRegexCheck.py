import re

re.sub(r'.*', "x", "a") # Noncompliant {{MESSAGE}}
re.sub(repl='x', pattern=r'.*', string='yyyy') # Noncompliant
re.subn(r'.*', "x", "a") # Noncompliant
re.compile(r'.*') # Noncompliant
re.search(r'.*', "foo") # Noncompliant
re.match(r'.*', "foo") # Noncompliant
re.fullmatch(r'.*', "foo") # Noncompliant
re.split(r'.*', "foo") # Noncompliant
re.findall(r'.*', "foo") # Noncompliant
re.finditer('.*', "foo") # Noncompliant

some_pattern = '.*' # Noncompliant
re.match(some_pattern, 'some string')
re.match(some_unknown_pattern, 'some string')
re.match(get_pattern(), 'some string')

re.match('.*\N{GREEK SMALL LETTER FINAL SIGMA}', 'foo') # We do ignore not raw strings containing \N escape sequences
re.match(r'.*\N{GREEK SMALL LETTER FINAL SIGMA}', 'foo') # Noncompliant

some_var = 'foo'
re.match(f'.*{some_var}', 'foo') # We do ignore f-strings that do contain an expression

re.sub(r'.*' r'.*', "x", "a") # We do not look at concats for now
re.sub() # Required arguments not provided
re.not_relevant_method(r'.*', "x", "a")
not_re.sub(r'.*', "x", "a")
