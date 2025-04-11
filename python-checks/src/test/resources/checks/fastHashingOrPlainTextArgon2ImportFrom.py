from argon2 import PasswordHasher
from argon2.profiles import CHEAPEST  # Noncompliant {{Use a secure Argon2 profile.}}

#                           ^^^^^^^^@-1
some_var = CHEAPEST  # Noncompliant {{Use a secure Argon2 profile.}}
#          ^^^^^^^^

foo(some_var)  # Noncompliant
#   ^^^^^^^^
