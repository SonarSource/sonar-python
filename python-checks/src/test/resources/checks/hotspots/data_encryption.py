############################################
###       pyca/cryptography library      ###
############################################

def pyca_cryptography():
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM, AESCCM
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, dsa, dh
    from cryptography.hazmat.primitives.ciphers import Cipher

    Fernet(key)                # Noncompliant
    ChaCha20Poly1305(key)      # Noncompliant
    AESGCM(key)                # Noncompliant
    AESCCM(key)                # Noncompliant
    Other_encryption_method()  # OK

    parameters_dh = dh.generate_parameters(2, key_size, backend) # Noncompliant {{Make sure that encrypting data is safe here.}}
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Cipher(algorithm, mode, backend)  # Noncompliant

############################################
###       pyca/pynacl library            ###
############################################

def pyca_cryptography():
    from nacl.public import Box
    from nacl.secret import SecretBox

    Box(secret_key, public_key)  # Noncompliant
    SecretBox(key)  # Noncompliant

############################################
###       PyCrypto library               ###
############################################

def pyCrypto():
    import Crypto
    from Crypto.Cipher import AES, DES, DES3, ARC2, ARC4, Blowfish, CAST, PKCS1_v1_5, PKCS1_OAEP, XOR
    from Crypto.PublicKey import ElGamal

    Crypto.Cipher.AES.new()       # Noncompliant
    Crypto.Other.AES.new()        # OK
    AES.new(key=key)              # Noncompliant
    DES.new(key=key)              # Noncompliant
    DES3.new(key=key)             # Noncompliant
    ARC2.new(key=key)             # Noncompliant
    ARC4.new(key=key)             # Noncompliant
    Blowfish.new(key=key)         # Noncompliant
    CAST.new(key=key)             # Noncompliant
    PKCS1_v1_5.new(key=key)       # Noncompliant
    PKCS1_OAEP.new(key=key)       # Noncompliant
    XOR.new(key=key)              # Noncompliant

    ElGamal.generate(key_size)    # Noncompliant


############################################
###       Cryptodome library             ###
############################################

def pyCrypto():
    import Cryptodome
    from Cryptodome.Cipher import AES, ChaCha20, DES, DES3, ARC2, ARC4, Blowfish, CAST, PKCS1_v1_5, PKCS1_OAEP, ChaCha20_Poly1305, Salsa20
    from Cryptodome.PublicKey import ElGamal

    Cryptodome.Cipher.AES.new()    # Noncompliant
    Cryptodome.Other.AES.new()     # OK
    AES.new(key=key)               # Noncompliant
    ChaCha20.new(key=key)          # Noncompliant
    DES.new(key=key)               # Noncompliant
    DES3.new(key=key)              # Noncompliant
    ARC2.new(key=key)              # Noncompliant
    ARC4.new(key=key)              # Noncompliant
    Blowfish.new(key=key)          # Noncompliant
    CAST.new(key=key)              # Noncompliant
    PKCS1_v1_5.new(key=key)        # Noncompliant
    PKCS1_OAEP.new(key=key)        # Noncompliant
    ChaCha20_Poly1305.new(key=key) # Noncompliant
    Salsa20.new(key=key)           # Noncompliant

    ElGamal.generate(key_size)     # Noncompliant
