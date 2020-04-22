def pycryptodomexExamples():
  from Cryptodome.Cipher import DES, DES3, ARC2, ARC4, Blowfish, AES
  from Cryptodome.Random import get_random_bytes
  
  key = b'-8B key-'
  DES.new(key, DES.MODE_OFB) # Noncompliant {{DES works with 56-bit keys that allow attacks via exhaustive search.}}
# ^^^^^^^

  key = DES3.adjust_key_parity(get_random_bytes(24))
  cipher = DES3.new(key, DES3.MODE_CFB) # Noncompliant {{Triple DES is vulnerable to meet-in-the-middle attacks.}}
  #        ^^^^^^^^

  key = b'Sixteen byte key'
  cipher = ARC2.new(key, ARC2.MODE_CFB) # Noncompliant {{RC2 is vulnerable to a related-key attack.}}
  #        ^^^^^^^^

  key = b'Very long and confidential key'
  cipher = ARC4.new(key) # Noncompliant {{RC4 is vulnerable to several attacks.}}
  #        ^^^^^^^^

  key = b'An arbitrarily long key'
  cipher = Blowfish.new(key, Blowfish.MODE_CBC) # Noncompliant {{Blowfish uses a 64-bit block size, which makes it vulnerable to birthday attacks.}}
  #        ^^^^^^^^^^^^

  key = b'Sixteen byte key'
  cipher = AES.new(key, AES.MODE_CCM) # Compliant

  cipher = UnknownFlyingValue.new(key, UnknownMode.CBC) # Compliant, doesn't matter

  # Force the engine to generate an ambiguous symbol, for code coverage only.
  ambiguous = "" if 42 * 42 < 1700 else (lambda x: x * x)
  cipher = ambiguous.new(key, Unknown.Mode)


# pycryptodome is a drop-in replacement for pycrpypto, currently those two libraries are not differentiated
def pycroptodomeExamples():
  from Crypto.Cipher import DES, DES3, ARC2, ARC4, Blowfish, AES
  from Crypto.Random import get_random_bytes

  key = b'-8B key-'
  DES.new(key, DES.MODE_OFB) # Noncompliant {{DES works with 56-bit keys that allow attacks via exhaustive search.}}
# ^^^^^^^

  key = DES3.adjust_key_parity(get_random_bytes(24))
  cipher = DES3.new(key, DES3.MODE_CFB) # Noncompliant {{Triple DES is vulnerable to meet-in-the-middle attacks.}}
  #        ^^^^^^^^
  key = b'Sixteen byte key'
  cipher = ARC2.new(key, ARC2.MODE_CFB) # Noncompliant {{RC2 is vulnerable to a related-key attack.}}
  #        ^^^^^^^^
  key = b'Very long and confidential key'
  cipher = ARC4.new(key) # Noncompliant {{RC4 is vulnerable to several attacks.}}
  #        ^^^^^^^^
  key = b'An arbitrarily long key'
  cipher = Blowfish.new(key, Blowfish.MODE_CBC) # Noncompliant {{Blowfish uses a 64-bit block size, which makes it vulnerable to birthday attacks.}}
  #        ^^^^^^^^^^^^

def pycaExamples():
  import os
  from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
  from cryptography.hazmat.backends import default_backend

  key = os.urandom(16)
  iv = os.urandom(16)

  tdes4 = Cipher(algorithms.TripleDES(key), mode=None, backend=default_backend()) # Noncompliant {{Triple DES is vulnerable to meet-in-the-middle attacks.}}
  #              ^^^^^^^^^^^^^^^^^^^^
  bf3 = Cipher(algorithms.Blowfish(key), mode=None, backend=default_backend()) # Noncompliant {{Blowfish uses a 64-bit block size, which makes it vulnerable to birthday attacks.}}
  #            ^^^^^^^^^^^^^^^^^^^
  rc42 = Cipher(algorithms.ARC4(key), mode=None, backend=default_backend()) # Noncompliant {{RC4 is vulnerable to several attacks.}}
  #             ^^^^^^^^^^^^^^^

def pydesExamples():
  import pyDes;

  des1 = pyDes.des('ChangeIt')  # Noncompliant {{DES works with 56-bit keys that allow attacks via exhaustive search.}}
  #      ^^^^^^^^^
  des2 = pyDes.des('ChangeIt', pyDes.CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=pyDes.PAD_PKCS5) # Noncompliant {{DES works with 56-bit keys that allow attacks via exhaustive search.}}
  #      ^^^^^^^^^
  tdes1 = pyDes.triple_des('ChangeItWithYourKey!!!!!')  # Noncompliant {{Triple DES is vulnerable to meet-in-the-middle attacks.}}
  #       ^^^^^^^^^^^^^^^^
  tdes2 = pyDes.triple_des('ChangeItWithYourKey!!!!!', pyDes.CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=pyDes.PAD_PKCS5) # Noncompliant {{Triple DES is vulnerable to meet-in-the-middle attacks.}}
  #       ^^^^^^^^^^^^^^^^

def pycryptodomeCompliant():
  from Crypto.Cipher import AES
  key = b'Sixteen byte key'
  cipher = AES.new(key, AES.MODE_CCM) # Compliant

def pycaCompliant():
  import os
  from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
  from cryptography.hazmat.backends import default_backend
  key = os.urandom(16)
  iv = os.urandom(16)
  aes2 = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend()) # Compliant

def pycryptoCompliant():
  from Crypto.Cipher import *
  aes1 = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456') # Compliant

