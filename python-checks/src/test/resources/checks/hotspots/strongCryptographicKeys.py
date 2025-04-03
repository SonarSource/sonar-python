############################################
###            cryptography              ###
############################################
def cryptography():
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, dsa
    from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

    # key_size = DSA key size
    dsa.generate_private_key(key_size=1024) # Noncompliant
#                            ^^^^^^^^^^^^^
    dsa.generate_private_key(key_size=2048) # Compliant
    dsa.generate_private_key(key_size=key_length) # Compliant
    dsa.generate_private_key(999) # Noncompliant
    dsa.generate_private_key(65537, 1024) # Noncompliant
    rsa.generate_private_key(65537, 2048) # Compliant
    dsa.generate_private_key(key_size=1024L) # Noncompliant
    # Test non integer literal for coverage
    dsa.generate_private_key(key_size=3.14)
    rsa.generate_private_key(public_exponent=3.14, key_size=2048)

    rsa.generate_private_key()
    rsa.generate_private_key(*foo)

    # key_size = RSA key size
    # public_exponent = RSA public key exponent
    rsa.generate_private_key(public_exponent=65537, key_size=1024) # Noncompliant {{Use a key length of at least 2048 bits.}}
    rsa.generate_private_key(key_size=1024, public_exponent=65537) # Noncompliant {{Use a key length of at least 2048 bits.}}
    rsa.generate_private_key(public_exponent=999, key_size=2048)   # Noncompliant {{Use a public key exponent of at least 65537.}}
    generate_private_key(public_exponent=999, key_size=2048)   # Noncompliant {{Use a public key exponent of at least 65537.}}
    rsa.generate_private_key(public_exponent=999, key_size=1024) # Noncompliant {{Use a key length of at least 2048 bits.}} {{Use a public key exponent of at least 65537.}}
    rsa.generate_private_key(public_exponent=65537, key_size=2048) # Compliant
    rsa.generate_private_key(public_exponent=exp, key_size=2048) # Compliant
    foo(key_size=1024) # Compliant

    # curve = ECC predefined curve.
    # Forbidden values for curve parameter: SECP192R1, SECT163K1, SECT163R2
    private_key_ec = ec.generate_private_key(curve=ec.SECT163R2)  # Noncompliant {{Use a key length of at least 224 bits.}}
    ec.generate_private_key(ec.SECT163R2) # Noncompliant
    ec.generate_private_key(curve=ec.SECT409R1) # Compliant
    ec.generate_private_key(curve=ec.Other) # Compliant
    ec.generate_private_key(curve=other.SECT409R1) # Compliant
    ec.generate_private_key(curve=rsa.SECT409R1) # Compliant
    ec.generate_private_key(curve=c) # Compliant
    ec.generate_private_key(curve=getCurve().SECT163R2) # Compliant


############################################
###                Crypto                ###
############################################

def crypto():
    from Crypto.PublicKey import DSA, RSA

    # bits = DSA key size
    DSA.generate(bits=1024) # Noncompliant
    DSA.generate(bits=2048) # Compliant

    # bits = RSA key size
    # e = RSA public key exponent
    RSA.generate(bits=1024, e=65537) # Noncompliant
    RSA.generate(e=65537, bits=1024) # Noncompliant
    RSA.generate(bits=2048, e=999)   # Noncompliant
    RSA.generate(bits=2048, e=65537) # Compliant
    RSA.generate(2048, None, None, 999) # Noncompliant

############################################
###                Cryptodome            ###
############################################

def cryptodome():
    from Cryptodome.PublicKey import DSA, RSA, ElGamal, ECC

    # bits = DSA key size
    DSA.generate(bits=1024) # Noncompliant
    DSA.generate(bits=2048) # Compliant

    # bits = RSA key size
    # e = RSA public key exponent
    RSA.generate(bits=1024, e=65537) # Noncompliant
    RSA.generate(e=65537, bits=1024) # Noncompliant
    RSA.generate(bits=2048, e=999)   # Noncompliant
#                           ^^^^^
    RSA.generate(bits=2048, e=65537) # Compliant
    RSA.other(bits=2048, e=65537) # Compliant

    RSA.generate(2048, None, 999) # Noncompliant

    ElGamal.generate(1024)  # Noncompliant
    ElGamal.generate(2048)  # Compliant

    ECC.generate(curve="secp192r1")  # Noncompliant
    ECC.generate(curve="ed25519")  # Compliant
    ecc_algo = "p192"
    ECC.generate(ecc_algo)  # Noncompliant {{Use a NIST-approved elliptic curve.}}

############################################
###                pyOpenSSL             ###
############################################

def pyOpenSSL():
    from OpenSSL.crypto import PKey, TYPE_RSA, TYPE_DSA, TYPE_DH
    key_rsa1024 = PKey()
    key_rsa1024.generate_key(type=TYPE_RSA, bits=1024) # Noncompliant
    key_rsa1024.generate_key(TYPE_RSA, 2048)
    key_rsa1024.generate_key(TYPE_DSA, 1024) # Noncompliant
    key_rsa1024.generate_key(TYPE_DSA, 2048)
    key_rsa1024.generate_key(TYPE_DH, 1024)
