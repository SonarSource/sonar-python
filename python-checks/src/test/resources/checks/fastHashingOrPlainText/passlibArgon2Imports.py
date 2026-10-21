def implementation_import():
    from passlib.handlers.argon2 import argon2

    argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}
    argon2.using(time_cost=5, memory_cost=7168, parallelism=1)


def public_reexport():
    from passlib.hash import argon2

    argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}
    argon2.using(time_cost=5, memory_cost=7168, parallelism=1)


def aliases():
    from passlib.handlers.argon2 import argon2 as implementation_argon2
    from passlib.hash import argon2 as public_argon2

    implementation_argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}
    public_argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}
    using = implementation_argon2.using
    using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}


def module_imports():
    import passlib.hash
    import passlib.handlers.argon2

    passlib.hash.argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}
    passlib.handlers.argon2.argon2.using(time_cost=4, memory_cost=7167, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}


def unrelated_handler(argon2):
    argon2.using(time_cost=4, memory_cost=7167, parallelism=1)


def positional_parameters():
    from passlib.hash import argon2

    argon2.using("ID", 7167, 16, 4, parallelism=1)  # Noncompliant {{Use secure Argon2 parameters.}}
    argon2.using("ID", 7168, 16, 4, parallelism=1)
    argon2.using("ID", 7167, 16, 5, parallelism=1)
    argon2.using("ID", 7167, 16, 4, parallelism=2)
    # digest_size and checksum_size must not be mistaken for memory_cost and parallelism.
    argon2.using("ID", 7168, 16, 4, 32, 1)


class OtherHandler:
    @classmethod
    def using(cls, **kwargs):
        pass


OtherHandler.using(time_cost=4, memory_cost=7167, parallelism=1)
