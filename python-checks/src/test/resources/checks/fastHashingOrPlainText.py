## scrypt

def non_hashlib():
    import scrypt
    scrypt.hash(password, salt, r=2)  # Noncompliant {{Use strong scrypt parameters.}}
    #                           ^^^
    scrypt.hash(password, salt, r=8)
    scrypt.hash(password, salt, buflen=31)  # Noncompliant
    scrypt.hash(password, salt, buflen=32)
    scrypt.hash(password, salt, N=1 << 12)  # Noncompliant
    exp = 7
    scrypt.hash(password, salt, N=1 << exp)  # Noncompliant
    exp_2 = 1 << 3
    scrypt.hash(password, salt, N=exp_2)  # Noncompliant
    tmp = 7
    exp_3 = 1 << tmp
    scrypt.hash(password, salt, N=exp_3)  # Noncompliant
    scrypt.hash(password, salt, N=1 << 13)  # Compliant
    scrypt.hash(password, salt, 8192)  # Compliant
    scrypt.hash(password, salt, N=8191)  # Noncompliant


def hashlib():
    from hashlib import scrypt
    def hash_scrypt_1(password: bytes, salt: bytes) -> bytes:
        return scrypt(
            password,
            salt=salt,
            n=1 << 10,  # Noncompliant
            r=8,
            p=2,
            dklen=64,
        )

    def hash_scrypt_2(password: bytes, salt: bytes) -> bytes:
        return scrypt(
            password,
            salt=salt,
            n=1 << 17,
            r=4,  # Noncompliant
            p=1,
            dklen=64,
        )

    def hash_scrypt_3(password: bytes, salt: bytes) -> bytes:
        return scrypt(
            password,
            salt=salt,
            n=1 << 15,
            r=8,
            p=3,
            dklen=31,  # Noncompliant
        )

    def hash_scrypt_4(password: bytes, salt: bytes) -> bytes:
        return scrypt(
            password,
            salt=salt,
            n=1 << 14,
            r=8,
            p=5,
            dklen=64,
        )  # Compliant

    scrypt(password, salt, dklen=33)  # Compliant
    scrypt(password, salt, 1 << 12)  # Noncompliant
    scrypt(password, salt, 1 << 13)  # Compliant
    scrypt(password, salt, 1 << 32, 2)  # Noncompliant
    scrypt(password, salt, 1 << 32, 12, p, 0, 31)  # Noncompliant
    scrypt(password, salt, n=8192)
    scrypt(password, salt, n=8191)  # Noncompliant
    scrypt(password, salt, n=3 << 10)  # For coverage reason
    scrypt(password, salt, n=a << 14)  # For coverage reason


def cryptography():
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    Scrypt(
        salt,
        length,
        1 << 12,  # Noncompliant
        8,
        2,
    )
    Scrypt(
        salt,
        length,
        1 << 13,
        7,  # Noncompliant
        5,
    )
    Scrypt(
        salt,
        length,
        r=8  # Compliant
    )
    Scrypt(
        salt,
        32  # Compliant
    )
    Scrypt(
        salt,
        31  # Noncompliant
    )
    Scrypt(
        salt,
        length,
        n=8192
    )
    Scrypt(
        salt,
        length,
        n=8191  # Noncompliant
    )
    Scrypt(
        salt,
        length,
        n=8191,  # Noncompliant
        r=2  # Noncompliant
    )


def passlib():
    from passlib.hash import scrypt

    scrypt.using(salt, rounds=11)  # Noncompliant
    scrypt.using(salt, rounds=12)  # Compliant
    scrypt.using(salt, block_size=7)  # Noncompliant
    scrypt.using(salt, block_size=8)  # Compliant
    scrypt.using(salt, 16, 11)  # Noncompliant
    scrypt.using(salt, 16, 12, 7)  # Noncompliant


## PBKDF
def hashlib():
    from hashlib import pbkdf2_hmac

    pbkdf2_hmac(
        "sha1", password, salt, 10_000  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    pbkdf2_hmac(
        "sha1", password, salt, 1_300_000
    )
    pbkdf2_hmac(
        "sha256", password, salt, 10_000  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    pbkdf2_hmac(
        "sha256", password, salt, 600_000
    )
    pbkdf2_hmac(
        "sha512", password, salt, 10_000  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    pbkdf2_hmac(
        "sha512", password, salt, 210_000
    )
    algo = "sha1"
    pbkdf2_hmac(
        algo, password, salt, 10_000  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    iters = 10_000
    pbkdf2_hmac(
        algo, password, salt, iters  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    iters_2 = 1_300_000
    pbkdf2_hmac(
        algo, password, salt, iters_2
    )
    pbkdf2_hmac(
        "unknown_algo", password, salt, 999_999
    )
    pbkdf2_hmac(
        "sha1", password, salt  # Missing iteration count
    )
    pbkdf2_hmac(
        32, password, salt, 10_000
    )
    pbkdf2_hmac()


def cryptography():
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    PBKDF2HMAC(
        "sha1",
        length,
        salt,
        iterations=10_000,  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    PBKDF2HMAC(
        "sha1",
        length,
        salt,
        iterations=1_300_000
    )
    PBKDF2HMAC(
        "sha256",
        length,
        salt,
        10_000,  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    PBKDF2HMAC(
        "sha256",
        length,
        salt,
        600_000
    )
    PBKDF2HMAC(
        "sha512",
        length,
        salt,
        10_000,  # Noncompliant {{Use at least 100 000 iterations.}}
    )
    PBKDF2HMAC(
        "sha512",
        length,
        salt,
        210_000
    )


def passlib():
    from passlib.hash import pbkdf2_sha1, pbkdf2_sha256, pbkdf2_sha512

    pbkdf2_sha1.using(salt, rounds=10_000)  # Noncompliant {{Use at least 100 000 iterations.}}
    pbkdf2_sha1.using(salt, rounds=1_300_000)
    pbkdf2_sha256.using(salt, rounds=10_000)  # Noncompliant {{Use at least 100 000 iterations.}}
    pbkdf2_sha256.using(salt, rounds=600_000)
    pbkdf2_sha512.using(salt, rounds=10_000)  # Noncompliant {{Use at least 100 000 iterations.}}
    pbkdf2_sha512.using(salt, rounds=210_000)
    pbkdf2_sha1.using(salt)
    pbkdf2_sha256.using(salt)  # Noncompliant
    pbkdf2_sha512.using(salt)  # Noncompliant {{Use at least 100 000 iterations.}}
#   ^^^^^^^^^^^^^^^^^^^

## Argon2

def not_cheapest():
    from argon2 import PasswordHasher
    from argon2.profiles import RFC_9106_HIGH_MEMORY
    PasswordHasher.from_parameters(RFC_9106_HIGH_MEMORY)


def manual_unsafe():
    from argon2 import PasswordHasher, Parameters
    PasswordHasher(  # Noncompliant {{Use secure Argon2 parameters.}}
        time_cost=4,
        memory_cost=7167,
        parallelism=1,
    )
    PasswordHasher(
        time_cost=4,
        memory_cost=7167,
        parallelism=2,
    )
    PasswordHasher(
        time_cost=4,
        memory_cost=7167,
    )
    PasswordHasher(
        time_cost=4,
        parallelism=1,
    )
    PasswordHasher(
        memory_cost=7167,
        parallelism=1,
    )

    Parameters(type, version, salt_len, hash_len, 1, 1, 1)  # Noncompliant

    from argon2.low_level import hash_secret, hash_secret_raw
    hash_secret(  # Noncompliant
        password,
        salt,
        4,
        7167,
        1,
    )
    hash_secret(
        password,
        salt,
        6,
        7167,
        1,
    )

    hash_secret_raw(  # Noncompliant
        password,
        salt,
        4,
        7167,
        1,
    )
    hash_secret_raw(
        password,
        salt,
        6,
        7167,
        1,
    )

    from passlib.hash import argon2

    argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant
    argon2.using(time_cost=4, memory_cost=7167, parallelism=2)
