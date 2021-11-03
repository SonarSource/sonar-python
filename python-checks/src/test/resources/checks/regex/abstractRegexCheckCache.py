import re

some_pattern = '.*'
re.match(some_pattern, 'some text', re.M)
re.match(some_pattern, 'some text', re.I)