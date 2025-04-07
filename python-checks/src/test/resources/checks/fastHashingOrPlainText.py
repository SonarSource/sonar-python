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
