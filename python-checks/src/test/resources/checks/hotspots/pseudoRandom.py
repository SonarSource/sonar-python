import random
import os
from random import getrandbits

os.random()

session_token = None  # security keyword at module level

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

def test_random_class_obj_methods_password():
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

def test_random_pkg_methods_secret():
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


# No crypto import and no security keyword in scope
def random_delay_ms():
    random.randint(-100, 150)

def shuffle_bucket(bucket):
    random.shuffle(bucket)

def derive_private(x):  # "derive"/"private" must not match "iv"
    random.random()

def generate_password(n):
    def helper():
        return [random.random() for _ in range(n)]  # Noncompliant
    return helper

def outer(salt):
    return lambda: random.random()  # Noncompliant

def session_id_gen():
    random.random()  # Noncompliant

def make_random_bytes_thing():
    random.random()  # Noncompliant

class Token:
    token_value = random.random()  # Noncompliant
    def plain_scope(self):
        random.random()

def uses_nonce():
    nonce = random.random()  # Noncompliant

def aes256_key():
    random.random()  # Noncompliant
