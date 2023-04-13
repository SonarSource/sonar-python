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

def test_random_class_obj_methods():
    a = random.Random()
    a.random() # Noncompliant
    a.getrandbits() # Noncompliant
    a.randint() # Noncompliant
    a.sample() # Noncompliant
    a.choice() # Noncompliant
    a.choices() # Noncompliant
    a.randbytes() # Noncompliant
    a.randrange() # Noncompliant
    a.shuffle() # Noncompliant

def test_random_pkg_methods():
    random.random() # Noncompliant
    random.getrandbits() # Noncompliant
    random.randint() # Noncompliant
    random.sample() # Noncompliant
    random.choice() # Noncompliant
    random.choices() # Noncompliant
    random.randbytes() # Noncompliant
    random.randrange() # Noncompliant
    random.shuffle() # Noncompliant

def system_random():
    random.SystemRandom().randint(5, 25)
    r = random.SystemRandom()
    r.randint(5, 25)
