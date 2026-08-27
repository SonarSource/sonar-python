import ssl


def default_protocol_attributes():
    ssl.SSLContext()  # Noncompliant {{Use a stronger protocol, or upgrade to Python 3.10+ which uses secure defaults.}}
    ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # Noncompliant
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)  # Noncompliant
    ssl.SSLContext(ssl.PROTOCOL_TLS)  # Noncompliant

    secure_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    secure_context.minimum_version = ssl.TLSv1_2


def using_create_default_context():
    ssl.create_default_context()  # Noncompliant
    ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)  # Noncompliant
    ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)  # Noncompliant
    ssl.create_default_context(cafile="ca.pem")  # Noncompliant

    secure_context = ssl.create_default_context()
    secure_context.minimum_version = ssl.TLSv1_3


def setting_unsafe_maximum_version():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)  # Noncompliant {{Change this code to use a stronger protocol.}}
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    context.maximum_version = ssl.TLSVersion.TLSv1_1
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<
