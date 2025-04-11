PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.Argon2PasswordHasher",  # Compliant
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.BCryptSHA256PasswordHasher",
    "django.contrib.auth.hashers.ScryptPasswordHasher",
]

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.UnsaltedSHA1PasswordHasher",  # Noncompliant {{Use a secure hashing algorithm to store passwords.}}
]
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^@-1

NOT_A_HASHER = []

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.ScryptPasswordHasher",
    "django.contrib.auth.hashers.UnsaltedSHA1PasswordHasher"
]

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.CryptPasswordHasher",  # Noncompliant {{Use a secure hashing algorithm to store passwords.}}
    "django.contrib.auth.hashers.ScryptPasswordHasher",
]
