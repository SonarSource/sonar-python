import random
import os
from random import getrandbits

os.random()

random.random()  # Noncompliant
random.getrandbits(1) # Noncompliant
random.randint(0,9) # Noncompliant
random.sample(['a', 'b'], 1)  # Noncompliant
random.choice(['a', 'b'])  # Noncompliant
random.choices(['a', 'b'])  # Noncompliant
random.choice(['a', 'b'])() # Noncompliant

getrandbits
getrandbits() # Noncompliant
randint() # not imported, should not be resolved to random.randint

# Note: random.SystemRandom() is safe
sysrand = random.SystemRandom()
sysrand.getrandbits(1)
sysrand.randint(0,9)
sysrand.random()
