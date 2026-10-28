import ssl

class LegacySSLContext(ssl.SSLContext):
    def set_ciphersuites(self):
        pass
