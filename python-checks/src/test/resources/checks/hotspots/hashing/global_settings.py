PASSWORD_HASHERS=['django.contrib.auth.hashers.MD5PasswordHasher']  # Noncompliant
PASSWORD_HASHERS=[]  # OK
PASSWORD_HASHERS=['django.contrib.auth.hashers.PBKDF2PasswordHasher']
PASSWORD_HASHERS=['django.contrib.auth.hashers.PBKDF2PasswordHasher', 'django.contrib.auth.hashers.MD5PasswordHasher'] # Noncompliant
OTHER=['django.contrib.auth.hashers.MD5PasswordHasher']
