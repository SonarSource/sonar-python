import ssl
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

server = ThreadingHTTPServer(('0.0.0.0', 8443), SimpleHTTPRequestHandler)
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()  # Compliant, ssl.* call present at module top-level, same scope as the constructor call
