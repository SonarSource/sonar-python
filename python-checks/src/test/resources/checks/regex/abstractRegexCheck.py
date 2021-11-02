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
re.finditer(r'.*', "foo") # Noncompliant

re.sub(r'.*' r'.*', "x", "a") # We do not look at concats for now
re.sub() # Required arguments not provided
re.not_relevant_method(r'.*', "x", "a")
not_re.sub(r'.*', "x", "a")
