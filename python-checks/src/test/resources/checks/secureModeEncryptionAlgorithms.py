def pyca_tests():
  from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
  from cryptography.hazmat.primitives.asymmetric import rsa, padding

  Cipher(algorithms.TripleDES(key), modes.CBC(iv)) # Noncompliant {{Use secure mode and padding scheme.}}
#                                   ^^^^^^^^^^^^^
  Cipher(algorithms.Blowfish(key), modes.GCM(iv))  # Compliant
  Cipher()  # Compliant
  args = []
  Cipher(algorithms.Blowfish(key), *args)  # Compliant
  Cipher(algorithms.Blowfish(key), modes.ECB())  # Noncompliant
  Cipher(algorithms.AES(key), modes.CBC(iv))  # Noncompliant
  Cipher(mode = modes.CBC(iv), algorithm = algorithms.AES(key))  # Noncompliant
  Cipher(algorithms.AES(key), modes.OFB(iv))  # Compliant
  Cipher(algorithms.AES(key), modes.ECB())  # Noncompliant

  private_key = rsa.generate_private_key()
  public_key = private_key.public_key()
  ciphertext = public_key.encrypt(message, padding.OAEP())

  private_key.decrypt(ciphertext,padding.OAEP()) # Compliant
  public_key.encrypt(message, padding.PKCS1v15()) # Noncompliant
#                             ^^^^^^^^^^^^^^^^^^
  public_key.encrypt(padding = padding.PKCS1v15(), plaintext = message) # Noncompliant
  private_key.decrypt(ciphertext,padding.PKCS1v15())  # Noncompliant
  private_key.decrypt(padding = padding.PKCS1v15(), ciphertext = ciphertext)  # Noncompliant
  print(padding.PKCS1v15()) # OK

def pycrypto_tests():
  # https://pycrypto.readthedocs.io/en/latest/
  from Crypto.Cipher import DES, CAST, DES3, ARC2, Blowfish, AES, PKCS1_OAEP, PKCS1_v1_5

  DES.new(key, DES.MODE_ECB) # Noncompliant
  DES.new(mode = DES.MODE_ECB, key = key) # Noncompliant
  DES.new(key, DES.MODE_CBC, IV=iv)  # Noncompliant
  DES.new(key, DES.MODE_CFB, IV=iv)  # Compliant
  DES.new(key, DES.MODE_OFB, IV=iv)  # Compliant
  DES.new(key, DES.MODE_CTR, IV=iv, counter=ctr) # Compliant
  CAST.new(key, CAST.MODE_ECB)  # Noncompliant
  DES3.new(key, DES3.MODE_ECB)  # Noncompliant
  ARC2.new(key, ARC2.MODE_CBC, IV=iv)  # Noncompliant
  Blowfish.new(key, Blowfish.MODE_ECB)  # Noncompliant
  AES.new(key, AES.MODE_CBC, IV=iv)  # Noncompliant
  PKCS1_OAEP.new(key) # Compliant
  PKCS1_v1_5.new(key) # Noncompliant

def cryptodomex_test():
  from Cryptodome.Cipher import DES, PKCS1_v1_5

  DES.new(key, DES.MODE_ECB) # Noncompliant
  DES.new(mode = DES.MODE_ECB, key = key) # Noncompliant
  DES.new(key, DES.MODE_CBC)  # Noncompliant
  DES.new() # Compliant
  args = []
  DES.new(key, *args) # Compliant
  DES.new(key, DES.MODE_EAX)  # Compliant

  unknown.new(key, DES.MODE_ECB) # Compliant

  PKCS1_v1_5.new(key) # Noncompliant

def pydes_test():
  import pyDes
  pyDes.des('ChangeIt') # Noncompliant
# ^^^^^^^^^
