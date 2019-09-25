from OpenSSL import SSL
import ssl
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager

SSL.Context(SSL.SSLv2_METHOD)  # Noncompliant {{Change this code to use a stronger protocol.}}
#               ^^^^^^^^^^^^
# Keyword argument
SSL.Context(method=SSL.SSLv2_METHOD)  # Noncompliant
SSL.Context(SSL.TLSv1_2_METHOD)  # Compliant
SSL.Context(method=SSL.TLSv1_2_METHOD)  # Compliant

# ssl.SSLContext()
ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv2) # Noncompliant
ctx = ssl.SSLContext(protocol=ssl.PROTOCOL_SSLv2) # Noncompliant

# ssl.wrap_socket()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ssl.wrap_socket(s, ssl_version=ssl.PROTOCOL_SSLv2) # Noncompliant
#                                  ^^^^^^^^^^^^^^

# ssl.SSLContext()
ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2) # Compliant
ctx = ssl.SSLContext(protocol=ssl.PROTOCOL_TLSv1_2) # Compliant


# ssl.wrap_socket()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ssl.wrap_socket(s, ssl_version=ssl.PROTOCOL_TLSv1_2) # Compliant


class Ssl3Adapter(HTTPAdapter):
    """"Transport adapter that forces SSLv3"""

    def init_poolmanager(self, *pool_args, **pool_kwargs):

        self.poolmanager = PoolManager(
            *pool_args,
            ssl_version=ssl.PROTOCOL_SSLv3, # Noncompliant
            **pool_kwargs)

class Tls12Adapter(HTTPAdapter):
    """"Transport adapter that forces TLSv1.2"""

    def init_poolmanager(self, *pool_args, **pool_kwargs):
        self.poolmanager = PoolManager(
            *pool_args,
            ssl_version=ssl.PROTOCOL_TLSv1_2,
            **pool_kwargs)


class unrelated():
    someClass.S = toto
    PROTOCOL_SSLv2 = "someconstant"
    def met():
        foo(PROTOCOL_SSLv2) # compliant, symbol does not match qualified name
