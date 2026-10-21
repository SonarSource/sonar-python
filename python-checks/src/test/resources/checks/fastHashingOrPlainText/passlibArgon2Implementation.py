from passlib.handlers.argon2 import argon2

argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}
argon2.using(time_cost=5, memory_cost=7168, parallelism=1)
