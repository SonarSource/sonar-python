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
ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1) # Noncompliant
ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_1) # Noncompliant
ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2) # OK

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

def default_protocol_attributes():
    # Unsafe by default in Python versions under 3.10
    ssl.SSLContext() # Noncompliant {{Use a stronger protocol, or upgrade to Python 3.10+ which uses secure defaults.}}
    ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) # Noncompliant
    ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER) # Noncompliant
    ssl.SSLContext(ssl.PROTOCOL_TLS) # Noncompliant

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) # OK
    ctx.minimum_version = ssl.TLSv1_2  # Ensure only TLSv1.2 or higher is used

    secure_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    secure_ctx.minimum_version = ssl.TLSv1_3  # Only TLSv1.3 will be used

    unsafe_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) # Noncompliant {{Use a stronger protocol, or upgrade to Python 3.10+ which uses secure defaults.}}
    unsafe_ctx.minimum_version = ssl.TLSv1_1  # Minimum version set to TLSv1.1 is insecure

    # Reassignment FN: rule is not flow sensitive
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSv1_1

    # Invalid context assignment
    invalid_ctx[42] = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) # Noncompliant
    invalid_ctx.minimum_version = ssl.TLSv1_3


def setting_unsafe_maximum_version():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER) # Noncompliant {{Change this code to use a stronger protocol.}}
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ctx.maximum_version = ssl.TLSVersion.TLSv1_1
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^<

def disabling_unsafe_protocols_through_options():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    
    ctx.options |= ssl.OP_NO_SSLv2
    ctx.options |= ssl.OP_NO_SSLv3
    ctx.options |= ssl.OP_NO_TLSv1
    ctx.options |= ssl.OP_NO_TLSv1_1
    
    ctx2 = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # OK
    ctx2.options |= (ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1)

def incomplete_disabling_unsafe_protocols_through_options():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # Noncompliant

    ctx.options |= ssl.OP_NO_SSLv2
    ctx.options |= ssl.OP_NO_TLSv1
    ctx.options |= ssl.OP_NO_TLSv1_1

    ctx2 = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # Noncompliant
    ctx2.options |= (ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1)

def invalid_options_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # Noncompliant
    foo(ctx.options)
    ctx.options = ssl.OP_NO_SSLv2
    ctx.options |= get_options() # Possible FP (accepted)

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


def using_create_default_context():
    # Unsafe by default unless Python 3.10
    ctx = ssl.create_default_context()  # Noncompliant
    client_ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)  # Noncompliant
    server_ctx = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)  # Noncompliant
    ctx_with_ca = ssl.create_default_context(cafile="ca.pem")  # Noncompliant
    
    ctx_secure = ssl.create_default_context()  # Compliant
    ctx_secure.minimum_version = ssl.TLSv1_3
    
    ctx_unsafe = ssl.create_default_context()  # Noncompliant
    ctx_unsafe.minimum_version = ssl.TLSv1_1

    ctx_max_unsafe = ssl.create_default_context()  # Noncompliant
    ctx_max_unsafe.maximum_version = ssl.TLSv1_1

    # Default context with proper options
    ctx_options_secure = ssl.create_default_context()  # Compliant
    ctx_options_secure.options |= ssl.OP_NO_SSLv2
    ctx_options_secure.options |= ssl.OP_NO_SSLv3
    ctx_options_secure.options |= ssl.OP_NO_TLSv1
    ctx_options_secure.options |= ssl.OP_NO_TLSv1_1

    client_ctx = ssl.create_default_context(purpose=unknown())

def test_openssl_default_tls_methods():
    from OpenSSL import SSL
    
    # Default TLS methods are insecure by default
    ctx1 = SSL.Context(SSL.TLS_METHOD)  # Noncompliant
    ctx2 = SSL.Context(SSL.TLS_SERVER_METHOD)  # Noncompliant
    ctx3 = SSL.Context(method=SSL.TLS_CLIENT_METHOD)  # Noncompliant
    
    # Making contexts secure with set_min_proto_version
    ctx4 = SSL.Context(SSL.TLS_METHOD)  # Compliant
    ctx4.set_min_proto_version(SSL.TLS1_2_VERSION)
    
    ctx5 = SSL.Context(SSL.TLS_SERVER_METHOD)  # Compliant
    ctx5.set_min_proto_version(SSL.TLS1_3_VERSION)
    
    # Still insecure with lower versions
    ctx6 = SSL.Context(SSL.TLS_CLIENT_METHOD)  # Noncompliant
    ctx6.set_min_proto_version(SSL.TLS1_1_VERSION)
    
    # Making contexts secure with set_options
    ctx7 = SSL.Context(SSL.TLS_METHOD)  # Compliant
    ctx7.set_options(SSL.OP_NO_SSLv2 | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1)
    
    ctx8 = SSL.Context(SSL.TLS_SERVER_METHOD)  # Compliant
    ctx8.set_options(SSL.OP_NO_SSLv2)
    ctx8.set_options(SSL.OP_NO_SSLv3)
    ctx8.set_options(SSL.OP_NO_TLSv1)
    ctx8.set_options(SSL.OP_NO_TLSv1_1)
    
    # Incomplete set_options (missing OP_NO_TLSv1_1)
    ctx9 = SSL.Context(SSL.TLS_CLIENT_METHOD)  # Noncompliant
    ctx9.set_options(SSL.OP_NO_SSLv2 | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1)
    
    # Combination of methods
    ctx10 = SSL.Context(SSL.TLS_METHOD)  # Compliant
    ctx10.set_min_proto_version(SSL.TLS1_2_VERSION)
    ctx10.set_options(SSL.OP_NO_SSLv2 | SSL.OP_NO_SSLv3)
    
    # Invalid usage
    ctx11 = SSL.Context(SSL.TLS_SERVER_METHOD)  # Noncompliant
    foo(ctx11.set_options)
    
    # Reassignment (not flow-sensitive)
    ctx12 = SSL.Context(SSL.TLS_CLIENT_METHOD)
    ctx12.set_min_proto_version(SSL.TLS1_2_VERSION)
    ctx12 = SSL.Context(SSL.TLS_CLIENT_METHOD)  # FN

    ctx13 = SSL.Context(SSL.TLS_SERVER_METHOD) # Noncompliant
    ctx13.set_options()

    # Context used directly
    foo(SSL.Context(SSL.TLS_SERVER_METHOD)) # Noncompliant

    ctx14 = SSL.Context(SSL.TLS_SERVER_METHOD) # Noncompliant
    # Possible FP if ctx14 is configured in bar
    bar(ctx14)
