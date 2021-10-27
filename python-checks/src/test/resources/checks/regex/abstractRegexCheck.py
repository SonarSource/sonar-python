import re

re.sub(r'.*', "x", "a")
re.sub('.*', "x", "a") # We only look at raw strings for now
re.sub(r'.*' r'.*', "x", "a") # We do not look at concats for now
re.sub() # Required arguments not provided
re.not_relevant_method(r'.*', "x", "a")
not_re.sub(r'.*', "x", "a")
