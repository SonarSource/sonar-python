import ssl

# An optional import disables reaching definitions for the module scope.
try:
    import optional_dependency
except ImportError:
    pass

ORIGINAL_CIPHERS: str = "RC4"
CIPHERS = ORIGINAL_CIPHERS
CHANGING_CIPHERS = "RC4"
CHANGING_CIPHERS = "HIGH"
unpacked, other = "RC4", "RC4"

ctx = ssl.create_default_context()
ctx.set_ciphers(CIPHERS)  # Noncompliant
ctx.set_ciphers(CHANGING_CIPHERS)
ctx.set_ciphers(unpacked)


def outer_ciphers():
    ctx = ssl.create_default_context()
    ctx.set_ciphers(CIPHERS)  # Noncompliant
