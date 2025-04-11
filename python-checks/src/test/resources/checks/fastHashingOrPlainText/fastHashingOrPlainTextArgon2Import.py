from argon2 import PasswordHasher
import argon2.profiles

some_var = argon2.profiles.CHEAPEST  # Noncompliant {{Use a secure Argon2 profile.}}

#                          ^^^^^^^^@-1

foo(some_var)  # Noncompliant
#   ^^^^^^^^
