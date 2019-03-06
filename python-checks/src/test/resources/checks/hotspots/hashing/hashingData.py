############################################
###       Hazmat cryptography library    ###
############################################

# https://cryptography.io/en/latest/hazmat/primitives/cryptographic-hashes

from cryptography.hazmat.primitives import hashes


def my_hash(algorithm):
    hashes.Hash(algorithm)  # Noncompliant {{Make sure that hashing data is safe here.}}
#   ^^^^^^^^^^^

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

# Changing default hashers

from django.conf import settings

def update_settings(value):
    settings.PASSWORD_HASHERS = value  # Noncompliant, and also a bad practice
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    (settings.PASSWORD_HASHERS) = value  # Noncompliant
    mySettings.PASSWORD_HASHERS = value # OK
    foo.bar, settings.PASSWORD_HASHERS = value # Noncompliant
    settings.OTHER = value  # OK

from django.contrib.auth.hashers import make_password

# Calling make_password with a specific hasher name or salt should be reviewed
def my_make_password(password, salt, hasher):
    make_password(password, salt=salt)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
