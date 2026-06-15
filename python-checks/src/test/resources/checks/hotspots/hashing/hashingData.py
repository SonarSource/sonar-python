############################################
###       Hazmat cryptography library    ###
############################################

# https://cryptography.io/en/latest/hazmat/primitives/cryptographic-hashes

from cryptography.hazmat.primitives import hashes


def my_hash(algorithm):
    hashes.Hash(algorithm)  # OK
    hashes.MD5() # Noncompliant
    hashes.SHA1() # Noncompliant
    hashes.SHA256()
    hashes.SHA3_256()
    foo(hashes) #coverage

hashes #coverage
############################################
###                Django                ###
############################################

# https://passlib.readthedocs.io/en/stable/lib/passlib.hash.html

from django.contrib.auth.hashers import SHA1PasswordHasher

# Creating custom Hasher

class MyPBKDF2PasswordHasher(SHA1PasswordHasher):  # Noncompliant
#                            ^^^^^^^^^^^^^^^^^^
    pass

class MyPBKDF2PasswordHasher2(OtherPBKDF2PasswordHasher): # OK
    pass

class MyPBKDF2PasswordHasher2(getHasher()): # OK
    pass

# Changing default hashers

from django.conf import settings

def update_settings(other):
    value = [
      'django.contrib.auth.hashers.SHA1PasswordHasher',
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
      'django.contrib.auth.hashers.MD5PasswordHasher',
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
      'django.contrib.auth.hashers.UnsaltedSHA1PasswordHasher',
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
      'django.contrib.auth.hashers.UnsaltedMD5PasswordHasher',
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
      'django.contrib.auth.hashers.CryptPasswordHasher',
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
      'django.contrib.auth.hashers.PBKDF2PasswordHasher', # Compliant
      'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher', # Compliant
      'django.contrib.auth.hashers.Argon2PasswordHasher', # Compliant
      'django.contrib.auth.hashers.BCryptSHA256PasswordHasher', # Compliant
    ]
    settings.PASSWORD_HASHERS = value  # Noncompliant [[and also a bad practice]]
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    (settings.PASSWORD_HASHERS) = value  # Noncompliant
    settings.PASSWORD_HASHERS = ['django.contrib.auth.hashers.UnsaltedMD5PasswordHasher']  # Noncompliant
    unsalted = 'django.contrib.auth.hashers.UnsaltedMD5PasswordHasher'
    settings.PASSWORD_HASHERS = [unsalted] # Noncompliant
    settings.PASSWORD_HASHERS = [other]
    settings.PASSWORD_HASHERS = [getValue()]
    settings.PASSWORD_HASHERS = ['django.contrib.auth.hashers.BCryptSHA256PasswordHasher']
    settings.PASSWORD_HASHERS = other
    settings.PASSWORD_HASHERS = getValue()
    mySettings.PASSWORD_HASHERS = value # OK
    foo.bar, settings.PASSWORD_HASHERS = value # Noncompliant
    settings.OTHER = value  # OK

from django.contrib.auth.hashers import make_password

# Calling make_password with a specific hasher name should be reviewed
def my_make_password(password, salt, hasher):
    make_password(password, salt=salt)
    make_password(password, hasher=hasher)
    make_password(password, salt=salt, hasher="other")

    make_password(password, salt=salt, hasher='sha1') # Noncompliant
    h='sha1'
    make_password(password, salt=salt, hasher=h) # Noncompliant
    make_password(password, salt=salt, hasher=getHasher())

    # No issue is raised when only the password is provided, then only the configuration should be reviewed
    make_password(password)  # OK


############################################
###                Werkzeug              ###
############################################

from werkzeug.security import generate_password_hash

def hash_password(password, method):
    generate_password_hash(password)  # OK
    generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)
    generate_password_hash(password, method='md5', salt_length=8) # Noncompliant
    generate_password_hash(password, method='sha224', salt_length=8) # Noncompliant
    generate_password_hash(password, method=getMethod(), salt_length=8)
    generate_password_hash(password, method=method, salt_length=8)

############################################
###                Hashlib               ###
############################################

# https://docs.python.org/3/library/hashlib.html

import hashlib
from hashlib import md5

def hash_data(algorithm):
    hashlib.new(algorithm)  # OK

    hashlib.md5  # Noncompliant
    hashlib.md5(usedforsecurity=False)
    hashlib.md5(usedforsecurity=True) #Noncompliant
    is_security = True
    hashlib.md5(usedforsecurity=is_security) #Noncompliant

    # FP, looking at sourcegraph, a literal is used as the value of "usedforsecurity"
    is_security2 = False
    hashlib.md5(usedforsecurity=is_security2) #Noncompliant

    hashlib.sha1() # Noncompliant
    hashlib.sha224() # Noncompliant

    alg = 'md5'
    hashlib.new(alg)  # Noncompliant
    hashlib.new(alg, usedforsecurity=True)  # Noncompliant
    hashlib.new(alg, usedforsecurity=False)

    md5  # Noncompliant

# S4790 suppressed when the hashlib call (or its .hexdigest()/.digest() qualifier)
# is immediately consumed by a bounded slice — truncated output cannot serve
# as a password hash, MAC, or integrity digest.
def hash_for_identifier(payload, email, seed, n):
    uid    = hashlib.sha1(b"data").hexdigest()[:8]          # OK — short identifier
    short  = hashlib.md5(payload).digest()[:4]              # OK — bounded keyspace
    prefix = hashlib.sha1(email.encode()).hexdigest()[0:5]  # OK — k-anonymity prefix
    otp    = hashlib.sha1(seed).hexdigest()[-6:]            # OK — HOTP-style truncation
    direct = hashlib.md5(payload)[:8]                       # OK — direct slice without digest call
    s224   = hashlib.sha224(b"data").hexdigest()[:8]        # OK — sha224 with bounded slice

    # Still flagged: slice bound too large (could be full-digest use)
    hashlib.sha1(b"data").hexdigest()[:]        # Noncompliant
    hashlib.sha1(b"data").hexdigest()[:17]      # Noncompliant
    hashlib.sha1(b"data").hexdigest()[-17:]     # Noncompliant

    # Still flagged: variable or dynamic bound — not a literal
    hashlib.sha1(b"data").hexdigest()[:n]       # Noncompliant

    # Still flagged: stride present — bounded size not guaranteed
    hashlib.sha1(b"data").hexdigest()[-6::2]    # Noncompliant

    # Still flagged: positive lower bound with no upper — unbounded tail
    hashlib.sha1(b"data").hexdigest()[6:]       # Noncompliant

    # Still flagged: zero upper bound
    hashlib.sha1(b"data").hexdigest()[:0]       # Noncompliant

    # Still flagged: chained method is neither hexdigest nor digest
    hashlib.sha1(b"data").copy()[:8]            # Noncompliant

    # Still flagged: hexdigest accessed as attribute, not called
    hashlib.sha1(b"data").hexdigest             # Noncompliant

    # Still flagged: negative upper bound keeps almost the whole digest
    hashlib.sha1(b"data").hexdigest()[:-4]      # Noncompliant

    # Still flagged: multi-dimensional slice
    hashlib.sha1(b"data").hexdigest()[1:4, 2:8] # Noncompliant

    # Still flagged: float literal upper bound
    hashlib.sha1(b"data").hexdigest()[:1.5]     # Noncompliant

    # Still flagged: float literal lower bound magnitude
    hashlib.sha1(b"data").hexdigest()[-1.5:]    # Noncompliant

    # Still flagged: unary plus lower bound
    hashlib.sha1(b"data").hexdigest()[+5:]      # Noncompliant

    # Still flagged: variable under unary minus
    hashlib.sha1(b"data").hexdigest()[-n:]      # Noncompliant

    # Still flagged: intermediate variable breaks the chain
    h = hashlib.sha1(b"data")                  # Noncompliant
    h.hexdigest()[:8]

def test_if_usedforsecurity_works_only_for_hashlib():
    import Cryptodome
    Cryptodome.Hash.MD2.new(usedforsecurity=False) # Noncompliant

    import passlib.hash
    passlib.hash.apr_md5_crypt(usedforsecurity=False)  # Noncompliant

############################################
###                Passlib               ###
############################################

# https://passlib.readthedocs.io/en/stable/lib/passlib.hash.html

from passlib.hash import apr_md5_crypt

import passlib.hash

passlib.hash.apr_md5_crypt  # Noncompliant

apr_md5_crypt  # Noncompliant


############################################
###                Cryptodome            ###
############################################

def cryptodome():
    import Cryptodome
    from Cryptodome.Hash import MD2
    from Cryptodome.Hash import MD4
    from Cryptodome.Hash import MD5
    from Cryptodome.Hash import SHA1
    from Cryptodome.Hash import SHA224
    from Cryptodome.Hash import SHA256
    from Cryptodome.Hash import SHA384
    from Cryptodome.Hash import SHA512
    from Cryptodome.Hash import HMAC

    Cryptodome.Hash.MD2.new() # Noncompliant
    MD2.new()                 # Noncompliant
    MD4.new()                 # Noncompliant
    MD5.new()                 # Noncompliant
    SHA1.new()                # Noncompliant
    SHA224.new()              # Noncompliant
    SHA256.new()              # OK
    SHA384.new()              # OK
    SHA512.new()              # OK
    HMAC.new(b"\x00")         # OK

############################################
###                PyCrypto              ###
############################################

def pycrypto():
    import Crypto
    from Crypto.Hash import MD2
    from Crypto.Hash import MD4
    from Crypto.Hash import MD5
    from Crypto.Hash import SHA1
    from Crypto.Hash import SHA224
    from Crypto.Hash import SHA256
    from Crypto.Hash import SHA384
    from Crypto.Hash import SHA512
    from Crypto.Hash import HMAC

    Crypto.Hash.MD2.new() # Noncompliant
    MD2.new() # Noncompliant
    MD4.new() # Noncompliant
    MD5.new() # Noncompliant
    SHA1.new() # Noncompliant
    SHA224.new() # Noncompliant
    SHA256.new() # OK
    SHA384.new() # OK
    SHA512.new() # OK
    HMAC.new(b"\x00") # OK
