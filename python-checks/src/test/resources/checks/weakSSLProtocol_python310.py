import ssl


def setting_unsafe_maximum_version():
    context1 = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER) # In Python 3.10, this is secure by default
    # Setting a maximum version to TLSv1.1 makes it insecure
    context1.maximum_version = ssl.TLSVersion.TLSv1_1 # Noncompliant {{Change this code to use a stronger protocol.}}

    context2 = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER) # OK
    context2.maximum_version = ssl.TLSVersion.TLSv1_3
    context3 = ssl.SSLContext()
    ssl.SSLContext()
