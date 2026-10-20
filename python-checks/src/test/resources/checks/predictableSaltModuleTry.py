import hashlib

# An optional import disables reaching definitions for the module scope.
try:
    import optional_dependency
except ImportError:
    pass

ORIGINAL_SALT: bytes = b"fixed"
SALT = ORIGINAL_SALT
CHANGING_SALT = b"fixed"
CHANGING_SALT = get_random_salt()
unpacked, other = b"fixed", b"fixed"

hashlib.pbkdf2_hmac("sha256", password, SALT, 1000)  # Noncompliant
hashlib.pbkdf2_hmac("sha256", password, CHANGING_SALT, 1000)
hashlib.pbkdf2_hmac("sha256", password, unpacked, 1000)


def outer_salt(password):
    hashlib.pbkdf2_hmac("sha256", password, SALT, 1000)  # Noncompliant
