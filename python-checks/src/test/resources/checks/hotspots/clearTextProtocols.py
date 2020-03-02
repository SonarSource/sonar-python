import telnetlib
from telnetlib import Telnet
import ftplib
from ftplib import FTP
import smtplib
import ssl

def clear_text_protocol():
  url = "http://" # Noncompliant {{Using http protocol is insecure. Use https instead}}
  #     ^^^^^^^^^
  url = "http://exemple.com" # Noncompliant
  url = "http://0001::1" # Noncompliant
  url = "http://dead:beef::1" # Noncompliant
  url = "http://::dead:beef:1" # Noncompliant
  url = "http://192.168.0.1" # Noncompliant
  url = "http://10.1.1.123" # Noncompliant
  url = "http://subdomain.exemple.com" # Noncompliant
  #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  url = "ftp://" # Noncompliant {{Using ftp protocol is insecure. Use sftp, scp or ftps instead}}
  url = "ftp://anonymous@exemple.com" # Noncompliant
  url = "telnet://" # Noncompliant
  url = "telnet://anonymous@exemple.com" # Noncompliant {{Using telnet protocol is insecure. Use ssh instead}}

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
  url = "http://::1" # Compliant
  url = "ftp://user@localhost" # Compliant

  # url without authority
  url = "http:///" # Compliant

  # Argument default value
  def download(url='ssh://exemple.com'): # Compliant
      print(url)


  cnx = telnetlib.Telnet("towel.blinkenlights.nl") # Noncompliant {{Using telnet protocol is insecure. Use ssh instead}}
  #     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  cnx = Telnet("towel.blinkenlights.nl") # Noncompliant
  cnx = ftplib.FTP("194.244.111.175") # Noncompliant  {{Using ftp protocol is insecure. Use sftp, scp or ftps instead}}
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

  url = "http://0:0:0:0:0:0:0:1" # Compliant
  url = "http://0000:0000:0000:0000:0000:0000:0000:0001" # Compliant
  url = "http://::1" # Compliant
  url = "http://0::1" # Compliant
  url = "http://0:0:0::1" # Compliant
  url = "http://0000::0001" # Compliant
  url = "http://0000:0:0000::0001" # Compliant
  url = "http://0000:0:0000::1" # Compliant
  url = "http://0::0:1" # Compliant
  url = "ftp://user@localhost" # Compliant

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
