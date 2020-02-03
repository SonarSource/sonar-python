import crypt
import base64
import os
import hashlib
import os

# crypt-salt
password = 'password123'
salt = crypt.mksalt(crypt.METHOD_SHA256)

hash = crypt.crypt(password)         # Noncompliant {{Add an unpredictable salt value to this hash.}}
#      ^^^^^^^^^^^
hash = crypt.crypt(password, "")     # Noncompliant {{Make this salt unpredictable.}}
#                            ^^
hash = crypt.crypt(password, salt)     # Compliant



# hashlib-salt
email = 'info@sonarsource'
password = 'password123'
salt = os.urandom(32)

hash = hashlib.pbkdf2_hmac('sha256', password, b'', 100000)        # Noncompliant {{Make this salt unpredictable.}}
hash = hashlib.pbkdf2_hmac('sha256', password, b'D8VxSmTZt2E2YV454mkqAY5e', 100000)    # Noncompliant {{Make this salt unpredictable.}}
hash = hashlib.pbkdf2_hmac('sha256', password, email, 100000)     # Noncompliant {{Make this salt unpredictable.}}
hash = hashlib.scrypt(password, n=1024, r=1, p=1, salt=b'') # Noncompliant {{Make this salt unpredictable.}}
#                                                 ^^^^^^^^

hash_ = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)     # Compliant
hash_ = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), unknown, 100000)     # Compliant
hash_ = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), *unpack, 100000)     # Compliant
def func():
  salt = "string_literal"
# ^^^^^^^^^^^^^^^^^^^^^^^>
  hash = hashlib.pbkdf2_hmac('sha256', password, salt, 100000)     # Noncompliant {{Make this salt unpredictable.}}
#                                                ^^^^

# cryptography-salt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

def derive_password(password, salt, backend):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'D8VxSmTZt2E2YV454mkqAY5e', # Noncompliant {{Make this salt unpredictable.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        iterations=100000,
        backend=backend
    )
    key = kdf.derive(password)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt, # Compliant
        iterations=100000,
        backend=backend
    )
    key = kdf.derive(password)

    salt_ = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt_,  # Compliant
        iterations=100000,
        backend=backend
    )
    key = kdf.derive(password)

salt = os.urandom(16)
backend = default_backend()
derive_password(b"my great password", salt, backend)


#cryptodome-salt

from Cryptodome.Protocol.KDF import PBKDF2, scrypt, bcrypt
from Crypto.Hash import SHA512
from Crypto.Random import get_random_bytes


def derive_password(password, salt):

    PBKDF2(password,
        b'D8VxSmTZt2E2YV454mkqAY5e', # Noncompliant {{Make this salt unpredictable.}}
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        32, count=100000
    )

    key = scrypt(password, b'D8VxSmTZt2E2YV454mkqAY5e', 32, N=2**14, r=8, p=1) # Noncompliant {{Make this salt unpredictable.}}
#                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    key = bcrypt(password, 12, b'D8VxSmTZt2E2YV45') # Noncompliant {{Make this salt unpredictable.}}
#                              ^^^^^^^^^^^^^^^^^^^


    PBKDF2(password, salt, # Compliant
        32, count=100000
    )

    salt_ = get_random_bytes(32)
    PBKDF2(password, salt_, # Compliant
        32, count=100000,
    )

    key = scrypt(password, salt_, 32, N=2**14, r=8, p=1) # Compliant

    salt_16 = get_random_bytes(16)
    key = bcrypt(password, 12, salt_16) # Compliant


salt = get_random_bytes(32)
password = b'my super secret'
derive_password(password, salt)

unknown.openUnknown(open(__file__).read())
