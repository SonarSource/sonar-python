try:
    import ssl
except ImportError:
    ssl = None

if ssl is not None:
    ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv2) # Noncompliant
