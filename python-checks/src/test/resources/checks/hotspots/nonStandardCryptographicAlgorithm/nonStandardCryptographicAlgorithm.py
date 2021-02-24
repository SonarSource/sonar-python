from django.contrib.auth.hashers import BasePasswordHasher


class UpperPasswordHasher(BasePasswordHasher):  # Noncompliant {{Make sure using a non-standard cryptographic algorithm is safe here.}}
#                         ^^^^^^^^^^^^^^^^^^
  ...

class OtherPasswordHasher:
  ...

class AnotherPasswordHasher(getPasswordHasher()):
  ...

class PasswordHasher(unknown):
  ...
