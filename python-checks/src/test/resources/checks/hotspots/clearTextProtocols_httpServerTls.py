import socketserver
import ssl
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from wsgiref.simple_server import make_server


def tls_wrapped_http_server_in_function():
    server = ThreadingHTTPServer(('0.0.0.0', 8443), SimpleHTTPRequestHandler)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()  # Compliant, ssl.* call present in the same function scope


class CustomHandler(socketserver.BaseRequestHandler):
    def handle(self):
        pass


class CustomProtocolServer(socketserver.ThreadingTCPServer):
    pass


def not_an_http_server():
    CustomProtocolServer(('127.0.0.1', 9000), CustomHandler).serve_forever()  # Compliant, not an HTTP server


def not_an_http_server_via_variable():
    server = CustomProtocolServer(('127.0.0.1', 9001), CustomHandler)
    server.serve_forever()  # Compliant, not an HTTP server


def plain_http_server_no_tls_in_its_own_scope():
    server = HTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
    server.serve_forever()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^


def unrelated_function_with_ssl_call():
    # An ssl.* call in a sibling function must not suppress the finding above:
    # "same scope" is the enclosing function, not the whole module.
    context = ssl.create_default_context()
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")


def wsgiref_wsgi_server_is_still_an_http_server():
    # wsgiref.simple_server.WSGIServer subclasses http.server.HTTPServer, no TLS here: still noncompliant.
    httpd = make_server("localhost", 8080, lambda environ, start_response: [])
    httpd.serve_forever()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^


class MyServer(HTTPServer):
    def run(self):
        # Unbound-method call on a real HTTPServer, no TLS: still noncompliant.
        HTTPServer.serve_forever(self)  # Noncompliant


class CustomProtocolServer2(socketserver.ThreadingTCPServer):
    def run(self):
        # Unbound-method call, still not an HTTP server: compliant.
        CustomProtocolServer2.serve_forever(self)


class MyServerViaGrandparent(HTTPServer):
    def run(self):
        # Unbound call qualified by an ancestor other than HTTPServer itself: the object that
        # actually serves is the argument (self, a real HTTPServer subclass instance), not the
        # qualifier - still noncompliant, no TLS.
        socketserver.TCPServer.serve_forever(self)  # Noncompliant


def chained_construction_no_intermediate_variable():
    # Receiver is directly the constructor call (no assignment to trace back to): still noncompliant, no TLS.
    HTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler).serve_forever()  # Noncompliant
