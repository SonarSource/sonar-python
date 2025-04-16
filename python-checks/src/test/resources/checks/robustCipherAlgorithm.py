def pycryptodomex_examples():
  from Cryptodome.Cipher import DES, DES3, ARC2, ARC4, Blowfish, AES, CAST
  from Cryptodome.Random import get_random_bytes

  key = b'-8B key-'
  DES.new(key, DES.MODE_OFB) # Noncompliant {{Use a strong cipher algorithm.}}
# ^^^^^^^

  key = DES3.adjust_key_parity(get_random_bytes(24))
  cipher = DES3.new(key, DES3.MODE_CFB) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^

  key = b'Sixteen byte key'
  cipher = ARC2.new(key, ARC2.MODE_CFB) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^

  key = b'Very long and confidential key'
  cipher = ARC4.new(key) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^

  key = b'An arbitrarily long key'
  cipher = Blowfish.new(key, Blowfish.MODE_CBC) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^^^^^

  key = os.urandom(16)
  cipher = CAST.new(key, CAST.MODE_OPENPGP)  # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^

  key = b'Sixteen byte key'
  cipher = AES.new(key, AES.MODE_CCM) # Compliant

  cipher = UnknownFlyingValue.new(key, UnknownMode.CBC) # Compliant, doesn't matter

  # Force the engine to generate an ambiguous symbol, for code coverage only.
  ambiguous = "" if 42 * 42 < 1700 else (lambda x: x * x)
  cipher = ambiguous.new(key, Unknown.Mode)


# pycryptodome is a drop-in replacement for pycrpypto, currently those two libraries are not differentiated
def pycroptodome_examples():
  from Crypto.Cipher import DES, DES3, ARC2, ARC4, Blowfish, AES, CAST, XOR
  from Crypto.Random import get_random_bytes

  key = b'-8B key-'
  DES.new(key, DES.MODE_OFB) # Noncompliant {{Use a strong cipher algorithm.}}
# ^^^^^^^

  key = DES3.adjust_key_parity(get_random_bytes(24))
  cipher = DES3.new(key, DES3.MODE_CFB) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^
  key = b'Sixteen byte key'
  cipher = ARC2.new(key, ARC2.MODE_CFB) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^
  key = b'Very long and confidential key'
  cipher = ARC4.new(key) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^
  key = b'An arbitrarily long key'
  cipher = Blowfish.new(key, Blowfish.MODE_CBC) # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^^^^^

  key = os.urandom(16)
  cipher = CAST.new(key, CAST.MODE_OPENPGP)  # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^^

  key = os.urandom(16)
  cipher = XOR.new(key)  # Noncompliant {{Use a strong cipher algorithm.}}
  #        ^^^^^^^

def pyca_examples():
  import os
  from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
  from cryptography.hazmat.decrepit.ciphers.algorithms import AnyAlgoFromDeprecitModule
  from cryptography.hazmat.backends import default_backend

  key = os.urandom(16)
  iv = os.urandom(16)

  tdes4 = Cipher(algorithms.TripleDES(key), mode=None, backend=default_backend()) # Noncompliant {{Use a strong cipher algorithm.}}
  #              ^^^^^^^^^^^^^^^^^^^^
  bf3 = Cipher(algorithms.Blowfish(key), mode=None, backend=default_backend()) # Noncompliant {{Use a strong cipher algorithm.}}
  #            ^^^^^^^^^^^^^^^^^^^
  rc42 = Cipher(algorithms.ARC4(key), mode=None, backend=default_backend()) # Noncompliant {{Use a strong cipher algorithm.}}
  #             ^^^^^^^^^^^^^^^
  casts5 = Cipher(algorithms.CAST5(key), mode=None, backend=default_backend())  # Noncompliant {{Use a strong cipher algorithm.}}
  #               ^^^^^^^^^^^^^^^^
  deprecit = Cipher(AnyAlgoFromDeprecitModule(key), mode=None, backend=default_backend())  # Noncompliant {{Use a strong cipher algorithm.}}
  #                 ^^^^^^^^^^^^^^^^^^^^^^^^^

def pydes_examples():
  import pyDes;

  des1 = pyDes.des('ChangeIt')  # Noncompliant {{Use a strong cipher algorithm.}}
  #      ^^^^^^^^^
  des2 = pyDes.des('ChangeIt', pyDes.CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=pyDes.PAD_PKCS5) # Noncompliant {{Use a strong cipher algorithm.}}
  #      ^^^^^^^^^
  tdes1 = pyDes.triple_des('ChangeItWithYourKey!!!!!')  # Noncompliant {{Use a strong cipher algorithm.}}
  #       ^^^^^^^^^^^^^^^^
  tdes2 = pyDes.triple_des('ChangeItWithYourKey!!!!!', pyDes.CBC, "\0\0\0\0\0\0\0\0", pad=None, padmode=pyDes.PAD_PKCS5) # Noncompliant {{Use a strong cipher algorithm.}}
  #       ^^^^^^^^^^^^^^^^

def pyssl_examples():
  import ssl
  ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
  ctx.set_ciphers('ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA:ECDHE-ECDSA-AES256-SHA:SRP-DSS-AES-256-CBC-SHA:SRP-RSA-AES-256-CBC-SHA:SRP-AES-256-CBC-SHA:DH-DSS-AES256-GCM-SHA384:DHE-DSS-AES256-GCM-SHA384:DH-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-SHA256:DHE-DSS-AES256-SHA256:DH-RSA-AES256-SHA256:DH-DSS-AES256-SHA256:DHE-RSA-AES256-SHA:DHE-DSS-AES256-SHA:DH-RSA-AES256-SHA:DH-DSS-AES256-SHA:DHE-RSA-CAMELLIA256-SHA:DHE-DSS-CAMELLIA256-SHA:DH-RSA-CAMELLIA256-SHA:DH-DSS-CAMELLIA256-SHA:ECDH-RSA-AES256-GCM-SHA384:ECDH-ECDSA-AES256-GCM-SHA384:ECDH-RSA-AES256-SHA384:ECDH-ECDSA-AES256-SHA384:ECDH-RSA-AES256-SHA:ECDH-ECDSA-AES256-SHA:AES256-GCM-SHA384:AES256-SHA256:AES256-SHA:CAMELLIA256-SHA:PSK-AES256-CBC-SHA:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-SHA256:ECDHE-ECDSA-AES128-SHA256:ECDHE-RSA-AES128-SHA:ECDHE-ECDSA-AES128-SHA:SRP-DSS-AES-128-CBC-SHA:SRP-RSA-AES-128-CBC-SHA:SRP-AES-128-CBC-SHA:DH-DSS-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:DH-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-SHA256:DHE-DSS-AES128-SHA256:DH-RSA-AES128-SHA256:DH-DSS-AES128-SHA256:DHE-RSA-AES128-SHA:DHE-DSS-AES128-SHA:DH-RSA-AES128-SHA:DH-DSS-AES128-SHA:DHE-RSA-SEED-SHA:DHE-DSS-SEED-SHA:DH-RSA-SEED-SHA:DH-DSS-SEED-SHA:DHE-RSA-CAMELLIA128-SHA:DHE-DSS-CAMELLIA128-SHA:DH-RSA-CAMELLIA128-SHA:DH-DSS-CAMELLIA128-SHA:ECDH-RSA-AES128-GCM-SHA256:ECDH-ECDSA-AES128-GCM-SHA256:ECDH-RSA-AES128-SHA256:ECDH-ECDSA-AES128-SHA256:ECDH-RSA-AES128-SHA:ECDH-ECDSA-AES128-SHA:AES128-GCM-SHA256:AES128-SHA256:AES128-SHA:SEED-SHA:CAMELLIA128-SHA:IDEA-CBC-SHA:PSK-AES128-CBC-SHA:ECDHE-RSA-RC4-SHA:ECDHE-ECDSA-RC4-SHA:ECDH-RSA-RC4-SHA:ECDH-ECDSA-RC4-SHA:RC4-SHA:RC4-MD5:PSK-RC4-SHA:ECDHE-RSA-DES-CBC3-SHA:ECDHE-ECDSA-DES-CBC3-SHA:SRP-DSS-3DES-EDE-CBC-SHA:SRP-RSA-3DES-EDE-CBC-SHA:SRP-3DES-EDE-CBC-SHA:EDH-RSA-DES-CBC3-SHA:EDH-DSS-DES-CBC3-SHA:DH-RSA-DES-CBC3-SHA:DH-DSS-DES-CBC3-SHA:ECDH-RSA-DES-CBC3-SHA:ECDH-ECDSA-DES-CBC3-SHA:DES-CBC3-SHA:PSK-3DES-EDE-CBC-SHA')  # Noncompliant
  ciphers = 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA:ECDHE-ECDSA-AES256-SHA:SRP-DSS-AES-256-CBC-SHA:SRP-RSA-AES-256-CBC-SHA:SRP-AES-256-CBC-SHA:DH-DSS-AES256-GCM-SHA384:DHE-DSS-AES256-GCM-SHA384:DH-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-SHA256:DHE-DSS-AES256-SHA256:DH-RSA-AES256-SHA256:DH-DSS-AES256-SHA256:DHE-RSA-AES256-SHA:DHE-DSS-AES256-SHA:DH-RSA-AES256-SHA:DH-DSS-AES256-SHA:DHE-RSA-CAMELLIA256-SHA:DHE-DSS-CAMELLIA256-SHA:DH-RSA-CAMELLIA256-SHA:DH-DSS-CAMELLIA256-SHA:ECDH-RSA-AES256-GCM-SHA384:ECDH-ECDSA-AES256-GCM-SHA384:ECDH-RSA-AES256-SHA384:ECDH-ECDSA-AES256-SHA384:ECDH-RSA-AES256-SHA:ECDH-ECDSA-AES256-SHA:AES256-GCM-SHA384:AES256-SHA256:AES256-SHA:CAMELLIA256-SHA:PSK-AES256-CBC-SHA:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-SHA256:ECDHE-ECDSA-AES128-SHA256:ECDHE-RSA-AES128-SHA:ECDHE-ECDSA-AES128-SHA:SRP-DSS-AES-128-CBC-SHA:SRP-RSA-AES-128-CBC-SHA:SRP-AES-128-CBC-SHA:DH-DSS-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:DH-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-SHA256:DHE-DSS-AES128-SHA256:DH-RSA-AES128-SHA256:DH-DSS-AES128-SHA256:DHE-RSA-AES128-SHA:DHE-DSS-AES128-SHA:DH-RSA-AES128-SHA:DH-DSS-AES128-SHA:DHE-RSA-SEED-SHA:DHE-DSS-SEED-SHA:DH-RSA-SEED-SHA:DH-DSS-SEED-SHA:DHE-RSA-CAMELLIA128-SHA:DHE-DSS-CAMELLIA128-SHA:DH-RSA-CAMELLIA128-SHA:DH-DSS-CAMELLIA128-SHA:ECDH-RSA-AES128-GCM-SHA256:ECDH-ECDSA-AES128-GCM-SHA256:ECDH-RSA-AES128-SHA256:ECDH-ECDSA-AES128-SHA256:ECDH-RSA-AES128-SHA:ECDH-ECDSA-AES128-SHA:AES128-GCM-SHA256:AES128-SHA256:AES128-SHA:SEED-SHA:CAMELLIA128-SHA:IDEA-CBC-SHA:PSK-AES128-CBC-SHA:ECDHE-RSA-RC4-SHA:ECDHE-ECDSA-RC4-SHA:ECDH-RSA-RC4-SHA:ECDH-ECDSA-RC4-SHA:RC4-SHA:RC4-MD5:PSK-RC4-SHA:ECDHE-RSA-DES-CBC3-SHA:ECDHE-ECDSA-DES-CBC3-SHA:SRP-DSS-3DES-EDE-CBC-SHA:SRP-RSA-3DES-EDE-CBC-SHA:SRP-3DES-EDE-CBC-SHA:EDH-RSA-DES-CBC3-SHA:EDH-DSS-DES-CBC3-SHA:DH-RSA-DES-CBC3-SHA:DH-DSS-DES-CBC3-SHA:ECDH-RSA-DES-CBC3-SHA:ECDH-ECDSA-DES-CBC3-SHA:DES-CBC3-SHA:PSK-3DES-EDE-CBC-SHA'
  ctx.set_ciphers(ciphers)  # Noncompliant
# ^^^^^^^^^^^^^^^
  ciphers2 = ciphers
  ctx.set_ciphers(ciphers2)  # Noncompliant
  ciphers3 = 'ECDHE-RSA-AES256-SHA'
  ctx.set_ciphers(ciphers3)  # Noncompliant
  ciphers4 = 'ECDHE:RSA:AES256:SHA'
  ctx.set_ciphers(ciphers4)  # Noncompliant
  ciphers5 = 'ECDHE:RSA:AES256:SHA:ECDHE-RSA-AES256-SHA'
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{The following cipher string is insecure: `SHA`}}
  ctx.set_ciphers(ciphers5)  # Noncompliant
# ^^^^^^^^^^^^^^^

  ctx = ssl.create_default_context()
  ctx.set_ciphers("NULL+SHA")  # Noncompliant

  ctx4 = ssl.create_default_context()
  ctx4.set_ciphers("DEFAULT:-RSA+LOW:!SHA:LOW")  # Noncompliant
# ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^< {{The following cipher string is insecure: `LOW`}}

  ctx5 = ssl.create_default_context()
  ctx5.set_ciphers("@SECLEVEL=1")  # Noncompliant

  ctx6 = ssl.create_default_context()
  ctx6.set_ciphers("@SECLEVEL=0")  # Noncompliant

  ciphers6 = 'ECDHE:RSA:AES256:LOW:ECDHE-RSA-AES256-SHA'
  #          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{The following cipher strings are insecure: `LOW`, `SHA`}}
  ctx.set_ciphers(ciphers6)  # Noncompliant
# ^^^^^^^^^^^^^^^

  context = ssl.create_default_context()
  context.set_ciphers("ECDH+3DES:DH+3DES:RSA+HIGH:RSA+3DES")  # Noncompliant

def py_open_ssl_examples():
    import socket
    from OpenSSL import SSL

    ctx = SSL.Context(SSL.TLS1_3_VERSION)
    ctx.set_cipher_list(b"@SECLEVEL=0")  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^

    ciphers2 = b'ECDHE:RSA:AES256:LOW:ECDHE-RSA-AES256-SHA'
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{The following cipher strings are insecure: `LOW`, `SHA`}}
    ctx.set_cipher_list(ciphers2)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^


def urllib3_ssl_context():
  import urllib3
  ctx = urllib3.util.ssl_.create_urllib3_context()
  ctx.set_ciphers("ECDH+3DES:DH+3DES:RSA+HIGH:RSA+3DES")  # Noncompliant

  urllib3.util.ssl_.create_urllib3_context(ciphers="ECDH+3DES:DH+3DES:RSA+HIGH:RSA+3DES")  # Noncompliant

def pycryptodome_compliant():
  from Crypto.Cipher import AES
  key = b'Sixteen byte key'
  cipher = AES.new(key, AES.MODE_CCM) # Compliant

def pyca_compliant():
  import os
  from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
  from cryptography.hazmat.backends import default_backend
  key = os.urandom(16)
  iv = os.urandom(16)
  aes2 = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend()) # Compliant

def pycrypto_compliant():
  from Crypto.Cipher import *
  aes1 = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456') # Compliant

def pyssl_compliant(unknown_cipher):
  import ssl
  ciphers = 'ECDHE-RSA-AES256-GCM-SHA384'
  ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
  ctx.set_ciphers(ciphers)
  ciphers2 = ''
  ctx.set_ciphers(ciphers2)
  ciphers3 = 'ECDHE:RSA:AES256:GCM:SHA384'
  ctx.set_ciphers(ciphers3)

  ciphers4 = 1
  ctx.set_ciphers(ciphers4)
  ctx.set_ciphers(unknown_cipher)
  ctx.set_ciphers(ciphers, ciphers2)

  ctx2 = ssl.create_default_context()
  ctx2.set_ciphers("ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:ECDH+AES128")  # Compliant

  ctx3 = ssl.create_default_context()
  ctx3.set_ciphers("DEFAULT:!eNULL:!aNULL:!MD5")  # Compliant
