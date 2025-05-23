from collections import defaultdict, namedtuple

d1 = defaultdict(default_factory=int) # Noncompliant {{Replace this keyword argument with a positional argument at the first place.}}
#                ^^^^^^^^^^^^^^^^^^^

d2 = defaultdict(int, default_factory=list) # Noncompliant {{Replace this keyword argument with a positional argument at the first place.}}
#                     ^^^^^^^^^^^^^^^^^^^^

d1 = defaultdict(int) # Compliant

named_tuple = namedtuple('Point', ['x', 'y']) # Compliant, for coverage
