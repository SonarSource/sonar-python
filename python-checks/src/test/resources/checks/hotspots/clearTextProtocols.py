import telnetlib
from telnetlib import Telnet
import ftplib
from ftplib import FTP
import smtplib
import ssl

def clear_text_protocol():
  url = "http://" # Noncompliant {{Using HTTP protocol is insecure. Use HTTPS instead.}}
  #     ^^^^^^^^^
  url = "http://exemple.com" # Noncompliant
  url = "http://0001::1" # Noncompliant
  url = "http://dead:beef::1" # Noncompliant
  url = "http://::dead:beef:1" # Noncompliant
  url = "http://192.168.0.1" # Noncompliant
  url = "http://10.1.1.123" # Noncompliant
  url = "http://subdomain.exemple.com" # Noncompliant
  #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  url = "ftp://" # Noncompliant {{Using FTP protocol is insecure. Use SFTP, SCP or FTPS instead.}}
  url = "ftp://anonymous@exemple.com" # Noncompliant
  url = "telnet://" # Noncompliant
  url = "telnet://anonymous@exemple.com" # Noncompliant {{Using Telnet protocol is insecure. Use SSH instead.}}

  # Argument default value
  def download(url='http://exemple.com'): # Noncompliant
      print(url)

  # Non sensitive url scheme
  url = "https://" # Compliant
  url = "sftp://" # Compliant
  url = "ftps://" # Compliant
  url = "scp://" # Compliant
  url = "ssh://" # Compliant

  # Only report string staring with the sensitive url scheme
  doc = "See http://exemple.com" # Compliant
  doc = "See ftp://exemple.com" # Compliant
  doc = "See telnet://exemple.com" # Compliant

  # The url domain component is a loopback address.
  url = "http://localhost" # Compliant
  url = "http://127.0.0.1" # Compliant
  url = "http://::1" # Noncompliant
  url = "ftp://user@localhost" # Compliant

  # url without authority
  url = "http:///" # Compliant

  # Argument default value
  def download(url='ssh://exemple.com'): # Compliant
      print(url)


  cnx = telnetlib.Telnet("towel.blinkenlights.nl") # Noncompliant {{Using Telnet protocol is insecure. Use SSH instead.}}
  #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  cnx = Telnet("towel.blinkenlights.nl") # Noncompliant
  cnx = ftplib.FTP("194.244.111.175") # Noncompliant  {{Using FTP protocol is insecure. Use SFTP, SCP or FTPS instead.}}
  cnx = FTP("194.244.111.175") # Noncompliant
  cnx = ftplib.FTP_TLS("secure.example.com") # Compliant
  cnx = FTP_TLS("secure.example.com") # Compliant

  # Exception: the url domain component is a loopback address.
  url = "http://localhost" # Compliant
  url = "http://LOCALHOST" # Compliant
  url = "http://127.0.0.1" # Compliant
  url = "http://127.0.0.1" # Compliant
  url = "http://127.0.0.001" # Compliant
  url = "http://127.0.00.1" # Compliant
  url = "http://127.00.0.1" # Compliant
  url = "http://127.000.000.001" # Compliant
  url = "http://127.0000.0000.1" # Compliant
  url = "http://127.0.01" # Compliant
  url = "http://127.1" # Compliant
  url = "http://127.001" # Compliant
  url = "http://127.0.0.254" # Compliant
  url = "http://127.63.31.15" # Compliant
  url = "http://127.255.255.254" # Compliant

  # Unbracketed IPv6 is not a valid HTTP URL (RFC 3986 requires brackets); the new filter no longer treats these as safe
  url = "http://0:0:0:0:0:0:0:1" # Noncompliant
  url = "http://0000:0000:0000:0000:0000:0000:0000:0001" # Noncompliant
  url = "http://::1" # Noncompliant
  url = "http://0::1" # Noncompliant
  url = "http://0:0:0::1" # Noncompliant
  url = "http://0000::0001" # Noncompliant
  url = "http://0000:0:0000::0001" # Noncompliant
  url = "http://0000:0:0000::1" # Noncompliant
  url = "http://0::0:1" # Noncompliant
  url = "ftp://user@localhost" # Compliant

  # Bracketed IPv6 loopback is a valid URL and is safe
  url = "http://[::1]" # Compliant
  url = "http://[0:0:0:0:0:0:0:1]" # Compliant
  url = "http://[::1]:8080/api" # Compliant

  # Cloud instance metadata endpoints are internal and not publicly reachable
  url = "http://169.254.169.254/latest/meta-data/" # Compliant
  url = "http://169.254.169.254/latest/api/token" # Compliant
  url = "http://[fd00:ec2::254]/latest/meta-data/" # Compliant
  url = "http://168.63.129.16/" # Compliant
  url = "http://100.100.100.200/latest/meta-data/" # Compliant
  url = "http://metadata.google.internal/computeMetadata/v1" # Compliant

  # Docker and Kubernetes internal hostnames are not publicly reachable
  url = "http://host.docker.internal:8085/metrics" # Compliant
  url = "http://vault.vault.svc.cluster.local:8200" # Compliant
  url = "http://auth-service.prod.svc.cluster.local:3001/auth" # Compliant

  # Well-known namespace URI authorities: http:// prefix is part of the identifier, not a network endpoint
  url = "http://www.w3.org/2001/XMLSchema" # Compliant
  url = "http://schemas.xmlsoap.org/soap/envelope/" # Compliant
  url = "http://www.springframework.org/schema/beans" # Compliant
  url = "http://maven.apache.org/POM/4.0.0" # Compliant
  url = "http://schema.org/Person" # Compliant

  # IANA-reserved documentation domains are placeholder URLs, not real endpoints
  url = "http://example.com/path" # Compliant
  url = "http://example.net" # Compliant
  url = "http://api.example.com/v1/users" # Compliant
  url = "ftp://example.com/file" # Compliant

  self.server_url = 'http://127.0.0.1:%s' % self.server.port # compliant, loopback
  def get_url(self, path):
          return "http://127.0.0.1:%d%s" % (self.port, path) # compliant, loopback

  data = self.urlopen("http://localhost:%s/" % handler.port) # compliant, loopback


  gravatar_url = u'http://www.gravatar.com/avatar/{0}?{1}'.format( # Noncompliant
      hashlib.md5(self.user.email.lower()).hexdigest(),
      urllib.urlencode({'d': no_picture, 's': '256'})
  )

  # Noncompliant@+1
  config = "http://cens.ioc.ee/projects/f2py2e/2.x"\
                                          "/F2PY-2-latest.tar.gz"
  url_in_multiline = ("the url is:"
                      "http://somedomain.com") # Noncompliant
  #                   ^^^^^^^^^^^^^^^^^^^^^^^
  url_in_multiline = ("the url is:"
                      "http://somedomain.com") # Noncompliant

  # SMTP lib
  smtp1 = smtplib.SMTP("smtp.gmail.com", port=587) # Noncompliant {{Make sure STARTTLS is used to upgrade to a secure connection using SSL/TLS.}}

  context = ssl.create_default_context()
  smtp2 = smtplib.SMTP("smtp.gmail.com", port=587) # Compliant
  smtp2.starttls(context=context)


  smtp3 = smtplib.SMTP_SSL("smtp.gmail.com", port=465) # Compliant

  smtp4 = smtplib.SMTP("smtp.gmail.com", port=587) # Noncompliant
  unknown.unknwon(smtp4)

def method(self):
  self.something[smth] = None

def ignore_multiple_assignments():
  smtp1 = smtp2 = smtplib.SMTP("smtp.gmail.com", port=587) # OK
  smtp2.starttls(context=context)

#FP
def FP_same_reference():
  smtp1 =  smtplib.SMTP("smtp.gmail.com", port=587) # Noncompliant
  smtp2 = smtp1
  smtp2.starttls(context=context)

#FP
def FP_starttls_in_different_method():
  smtp_safe =  smtplib.SMTP("smtp.gmail.com", port=587) # Noncompliant
  start_tls(smtp_safe)

def start_tls(x):
  x.starttls(context=context)


def python_web_server_noncompliant():
    from http.server import SimpleHTTPRequestHandler, HTTPServer, ThreadingHTTPServer, socketserver

    http_server = HTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
    http_server.serve_forever()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    threading_http_server = ThreadingHTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
    threading_http_server.serve_forever()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    class MyServer(HTTPServer):
        def run(self):
            HTTPServer.server_bind(self)  # Noncompliant
        #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            super(self).serve_forever()  # Noncompliant
        #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    my_server = MyServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
    my_server.serve_forever()  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^

    class MyThreadingServer(ThreadingHTTPServer):
        def run(self):
            super().server_bind()  # Noncompliant
        #   ^^^^^^^^^^^^^^^^^^^^^
            super(self).serve_forever()  # Noncompliant
        #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    my_threading_server = MyThreadingServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
    my_threading_server.serve_forever()  # Noncompliant

    import http.server
    class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        def server_bind(self):
            HTTPServer.server_bind(self) # Noncompliant


def python_web_server_compliant(ok_server):

    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from http.server import HTTPServer as PythonServer

    http_server = PythonServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
    http_server.server_bind()  # Compliant we do not raise here as the call to server bind was done already in the constructor

    class MyThreadingServer(ThreadingHTTPServer):
        def server_bind(self):  # Compliant
            pass

    ok_server.serve_forever()  # Compliant
    ok_server.server_bind()

    class HTTPServer():
        def serve_forever(self):
            pass

        def server_bind(self):
            pass

    class MyServer(HTTPServer):
        def run(self):
            super(self).serve_forever()  # Compliant

        def server_bind(self):
            HTTPServer.server_bind(self)

    my_server = MyServer()
    my_server.serve_forever()
    my_server.server_bind()

    my_server = HTTPServer()
    my_server.serve_forever()
    my_server.server_bind()
