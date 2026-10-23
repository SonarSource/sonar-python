import random
from cryptography.hazmat.primitives.ciphers import Cipher

def pick_color():
    return random.random()  # Noncompliant

sysrand = random.SystemRandom()
sysrand.random()
