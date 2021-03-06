import urllib.request
import ssl

# ValueError: Cannot set verify_mode to CERT_NONE when check_hostname is enabled

ssl._create_unverified_context() # OK
ctx1 = ssl._create_unverified_context() # Noncompliant

ctx_unverified_ok = ssl._create_unverified_context() # Compliant
ctx_unverified_ok.check_hostname = True

ctx2 = ssl._create_stdlib_context() # Noncompliant
if ctx2.check_hostname:
  pass

ctx3 = ssl.create_default_context() # Noncompliant
ctx3.check_hostname = False

ctx4 = ssl.create_default_context()
ctx4.check_hostname = True # Compliant

ctx5 = ssl._create_default_https_context() # Compliant

r1 = urllib.request.urlopen('https://151.101.0.223', context=ssl._create_unverified_context()) # Noncompliant
r1 = urllib.request.urlopen('https://151.101.0.223', ssl._create_unverified_context()) # Noncompliant
r1 = urllib.request.urlopen('https://151.101.0.223', ssl._create_default_https_context()) # OK

urllib.request.urlopen('https://151.101.0.223', context=ssl._create_unverified_context()) # Noncompliant
urllib.request.urlopen('https://151.101.0.223', ssl._create_unverified_context()) # Noncompliant
urllib.request.urlopen('https://151.101.0.223', ssl._create_default_https_context()) # OK

urllib.request.something_else('https://151.101.0.223', ssl._create_default_https_context()) # OK
unknown.unknown = ssl._create_default_https_context()
unknown.otherunknown = ssl._create_unverified_context() # FN
(smth, unknwon) = ssl._create_default_https_context()

ssl._create_unverified_context() # OK, unused
unknown_method(ssl._create_unverified_context()) # OK

ctx6 = ssl._create_unverified_context() # Noncompliant
ctx6.unrelated_property = True

ctx7 = ssl._create_unverified_context() # Noncompliant
if ctx7 == whatever:
  pass

if ssl._create_unverified_context() == whatever:
  pass
