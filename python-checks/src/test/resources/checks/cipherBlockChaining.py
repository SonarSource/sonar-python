from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def cryptodome():
  key = 'foo'.encode('utf-8')
  data = 'bar'.encode('utf-8')

  random_vector = get_random_bytes(AES.block_size)
  enc1 = AES.new(key, AES.MODE_CBC, random_vector)
  enc1.encrypt(pad(data, AES.block_size))  # Compliant

  static_vector = b'x' * AES.block_size
  enc2 = AES.new(key, AES.MODE_CBC, static_vector)  # 2nd location
  cipher_text = enc2.encrypt(pad(data, AES.block_size))  # Noncompliant
  unpad(enc2.decrypt(cipher_text), AES.block_size)  # Compliant

  enc3 = AES.new(key)
  cipher_text = enc3.encrypt(pad(data, AES.block_size))  # OK

  enc4 = AES.new(key, unknown(), static_vector)
  cipher_text = enc4.encrypt(pad(data, AES.block_size))  # OK

  AES.new(key, AES.MODE_CBC, static_vector) # OK
  AES.new(key, AES.MODE_CBC)

  unknown_vector = urandom(16)
  unknown_vector = b'x' * 16
  enc5 = AES.new(key, AES.MODE_CBC, unknown_vector)
  cipher_text = enc5.encrypt(pad(data, AES.block_size))  # OK

  enc6 = AES.new(key, AES.MODE_CBC, static_vector) + 42
  cipher_text = enc2.encrypt(pad(data, AES.block_size)) # OK

  enc7 = AES.new(key, unknown, static_vector)
  cipher_text = enc7.encrypt(pad(data, AES.block_size))  # OK

  enc8 = AES.new(key, AES.MODE_CTR, static_vector)
  cipher_text = enc8.encrypt(pad(data, AES.block_size))  # OK


from os import urandom
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def cryptography():
  key = urandom(32)

  random_vector = urandom(16)
  enc1 = Cipher(algorithms.AES(key), modes.CBC(random_vector))
  enc1.encryptor()  # Compliant

  static_vector = b'x' * 16
  enc2 = Cipher(algorithms.AES(key), modes.CBC(static_vector))
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
  enc2.encryptor()  # Noncompliant {{Use a dynamically-generated, random IV.}}
# ^^^^
  enc2.decryptor()  # Compliant

  enc3 = Cipher(algorithms.AES(key))
  enc3.encryptor()

  enc4 = Cipher(algorithms.AES(key), unknown())
  enc4.encryptor()

  enc5 = Cipher(algorithms.AES(key), unknown)
  enc5.encryptor()

  Cipher(algorithms.AES(key), modes.CBC(static_vector)) # OK

  Cipher(algorithms.AES(key), modes.CBC())

  enc6 = Cipher(algorithms.AES(key), modes.CTR(static_vector))
  enc6.encryptor()  # OK

  enc7 = Cipher(algorithms.AES(key), modes.CBC(static_vector)) + 42
  enc2.encryptor()  # OK

  enc8 = Cipher(algorithms.AES(key), modes.CBC(static_vector))
  some_call(enc8)

  x, y = Cipher(algorithms.AES(key), modes.CBC(static_vector))
  x.encryptor()  # OK

  enc9 = Cipher(algorithms.AES(key), modes.CBC(static_vector))
  enc9.decryptor()  # OK

  random_vector_2 = urandom(16) + 42
  enc10 = Cipher(algorithms.AES(key), modes.CBC(random_vector_2))
  enc10.encryptor()  # Compliant

def no_stack_overflow_on_mutual_assignments():
    static_vector = other_static
    other_static = static_vector
    enc2 = AES.new(key, AES.MODE_CBC, static_vector)
    cipher_text = enc2.encrypt(pad(data, AES.block_size))  # OK
