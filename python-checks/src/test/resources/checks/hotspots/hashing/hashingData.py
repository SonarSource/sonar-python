############################################
###       Hazmat cryptography library    ###
############################################

# https://cryptography.io/en/latest/hazmat/primitives/cryptographic-hashes

from cryptography.hazmat.primitives import hashes


def my_hash(algorithm):
    hashes.Hash(algorithm)  # Noncompliant {{Make sure that hashing data is safe here.}}
#   ^^^^^^^^^^^
    foo(hashes) #coverage
hashes #coverage
############################################
###                Django                ###
############################################

# https://passlib.readthedocs.io/en/stable/lib/passlib.hash.html

from django.contrib.auth.hashers import PBKDF2PasswordHasher

# Creating custom Hasher

class MyPBKDF2PasswordHasher(PBKDF2PasswordHasher):  # Noncompliant
#                            ^^^^^^^^^^^^^^^^^^^^
    pass

class MyPBKDF2PasswordHasher2(OtherPBKDF2PasswordHasher): # OK
    pass

class MyPBKDF2PasswordHasher2(getHasher()): # OK
    pass

# Changing default hashers

from django.conf import settings

def update_settings(value):
    settings.PASSWORD_HASHERS = value  # Noncompliant [[and also a bad practice]]
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    (settings.PASSWORD_HASHERS) = value  # Noncompliant
    mySettings.PASSWORD_HASHERS = value # OK
    foo.bar, settings.PASSWORD_HASHERS = value # Noncompliant
    settings.OTHER = value  # OK

from django.contrib.auth.hashers import make_password

# Calling make_password with a specific hasher name or salt should be reviewed
def my_make_password(password, salt, hasher):
    make_password(password, salt=salt)  # Noncompliant
#   ^^^^^^^^^^^^^
    make_password(password, hasher=hasher)  # Noncompliant
    make_password(password, salt=salt, hasher=hasher)  # Noncompliant

    # No issue is raised when only the password is provided, then only the configuration should be reviewed
    make_password(password)  # OK


############################################
###                Werkzeug              ###
############################################

from werkzeug.security import generate_password_hash

def hash_password(password):
    generate_password_hash(password)  # Noncompliant

############################################
###                Hashlib               ###
############################################

# https://docs.python.org/3/library/hashlib.html

import hashlib
from hashlib import blake2b

def hash_data(algorithm):
    hashlib.new(algorithm)  # Noncompliant

    hashlib.blake2b  # Noncompliant

    hashlib.new(algorithm)  # Noncompliant

    blake2b  # Noncompliant


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
    from Cryptodome.Hash import SHA
    from Cryptodome.Hash import SHA224
    from Cryptodome.Hash import SHA256
    from Cryptodome.Hash import SHA384
    from Cryptodome.Hash import SHA512
    from Cryptodome.Hash import HMAC

    Cryptodome.Hash.MD2.new() # Noncompliant
    MD2.new()                 # Noncompliant
    MD4.new()                 # Noncompliant
    MD5.new()                 # Noncompliant
    SHA.new()                 # Noncompliant
    SHA224.new()              # Noncompliant
    SHA256.new()              # Noncompliant
    SHA384.new()              # Noncompliant
    SHA512.new()              # Noncompliant
    HMAC.new(b"\x00")         # Noncompliant

############################################
###                PyCrypto              ###
############################################

def pycrypto():
    import Crypto
    from Crypto.Hash import MD2
    from Crypto.Hash import MD4
    from Crypto.Hash import MD5
    from Crypto.Hash import SHA
    from Crypto.Hash import SHA224
    from Crypto.Hash import SHA256
    from Crypto.Hash import SHA384
    from Crypto.Hash import SHA512
    from Crypto.Hash import HMAC

    Crypto.Hash.MD2.new() # Noncompliant
    MD2.new() # Noncompliant
    MD4.new() # Noncompliant
    MD5.new() # Noncompliant
    SHA.new() # Noncompliant
    SHA224.new() # Noncompliant
    SHA256.new() # Noncompliant
    SHA384.new() # Noncompliant
    SHA512.new() # Noncompliant
    HMAC.new(b"\x00") # Noncompliant
