def pyopensslTest():
  # Mutably borrowed from here:
  # https://github.com/SonarSource/security-expected-issues/blob/master/python/rules/vulnerabilities/\
  # RSPEC-4830%20Server%20certificates%20should%20be%20verified%20during%20SSL%E2%81%84TLS%20connections/\
  # pyopenssl-test.py

  from OpenSSL import SSL
  import sys, os, select, socket

  ctx1 = SSL.Context(SSL.TLSv1_2_METHOD)
  ctx1.set_verify(SSL.VERIFY_PEER, verify_callback) # Compliant
  ctx1.set_verify(SSL.VERIFY_PEER | SSL.VERIFY_FAIL_IF_NO_PEER_CERT, verify_callback) # Compliant
  ctx1.set_verify(SSL.VERIFY_PEER | SSL.VERIFY_FAIL_IF_NO_PEER_CERT | VERIFY_CLIENT_ONCE, verify_callback) # Compliant

  ctx = SSL.Context(SSL.TLSv1_2_METHOD)
  ctx.set_verify(SSL.VERIFY_NONE, verify_callback) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                  ^^^^^^^^^^^

  ctx.set_verify(SSL.VERIFY_FAIL_IF_NO_PEER_CERT | SSL.VERIFY_NONE | SSL.VERIFY_CLIENT_ONCE) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                    ^^^^^^^^^^^

  # Weird cases for code coverage only.
  ctxC1 = SSL.Context(SSL.TLSv1_2_METHOD)
  ctxC1.set_verify()
  kwargs = { 'something': True }
  ctxC1.set_verify(**kwargs)
  ctxC1.set_verify(noSSL.THIS_DOESNT_EXIST)

  def truthy_verify_callback_local(a):
      if a:
        return True
      else:
        return 1

  def verify_callback_local(a):
      if a:
        return False
      else:
        return 1

  def tuple_return_verify_callback_local(a, b):
      return a, b

  locally_defined_non_function_callback = ""
  ctx2 = SSL.Context(SSL.TLSv1_2_METHOD)
  ctx2.set_verify(SSL.VERIFY_PEER, truthy_verify_callback_local) # Noncompliant
  #                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  ctx3 = SSL.Context(SSL.TLSv1_2_METHOD)
  ctx3.set_verify(SSL.VERIFY_PEER, tuple_return_verify_callback_local) # Compliant
  ctx3.set_verify(SSL.VERIFY_PEER, verify_callback_local) # Compliant
  ctx3.set_verify(SSL.VERIFY_PEER, locally_defined_non_function_callback) # Compliant
  ctx3.set_verify(SSL.VERIFY_PEER, something.unknown) # Compliant

  ctx4 = SSL.Context(SSL.TLSv1_2_METHOD) # Noncompliant

  ctx5 = SSL.Context(SSL.TLSv1_2_METHOD)
  ctx5.use_privatekey(SSL.VERIFY_NONE) # Check that other random method call is not raised with wrong arguments
  ctx5.set_verify(SSL.VERIFY_PEER, verify_callback_local) # Compliant

def requestsTests():
  # Mutably borrowed from here:
  # https://github.com/SonarSource/security-expected-issues/blob/master/python/rules/vulnerabilities/\
  # RSPEC-4830%20Server%20certificates%20should%20be%20verified%20during%20SSL%E2%81%84TLS%20connections/\
  # requests-tests.py
  import requests

  requests.request('GET', 'https://example.domain', verify=False) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^^^^
  requests.request('GET', 'https://example.domain', verify='') # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=0) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^
  requests.request('GET', 'https://example.domain', verify=0.0) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^^
  requests.request('GET', 'https://example.domain', verify=0j) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify="") # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=b'') # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^^
  requests.request('GET', 'https://example.domain', verify=[]) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify={}) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=()) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=set()) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^^^^
  requests.request('GET', 'https://example.domain', verify=range(0)) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^^^^^^^
  requests.request(verify=False, method='GET', url='https://example.domain') # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                       ^^^^^


  kargs1 = {'verify': False} # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                   ^^^^^
  requests.request('GET', 'https://example.domain', **kargs1)
  #                                                   ^^^^^^ < 1


  kargs2 = {'method': 'GET', 'url': 'https://example.domain', 'verify': False} # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                                     ^^^^^
  requests.request(**kargs2)
  #                  ^^^^^^ < 1

  requests.get('https://example.domain', verify=False) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                             ^^^^^
  requests.post('https://example.domain', verify=False) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                              ^^^^^
  requests.options('https://example.domain', verify=False) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                 ^^^^^

  requests.request(method='GET', url='https://example.domain') # Compliant
  requests.request(method='GET', url='https://example.domain', verify=True) # Compliant
  requests.request('GET', 'https://example.domain', verify='/path/to/CAbundle') # Compliant
  requests.request(verify=True, method='GET', url='https://example.domain') # Compliant
  kargs = {'verify': True}
  requests.request('GET', 'https://example.domain', **kargs) # Compliant
  kargs = {'method': 'GET', 'url': 'https://example.domain', 'verify': True}
  requests.request(**kargs) # Compliant

  requests.head(url='https://example.domain') # Compliant
  requests.get(url='https://example.domain') # Compliant
  requests.post(url='https://example.domain') # Compliant
  requests.put(url='https://example.domain') # Compliant
  requests.patch(url='https://example.domain') # Compliant
  requests.delete(url='https://example.domain') # Compliant
  requests.options(url='https://example.domain') # Compliant
  requests.request('GET', 'https://example.domain', verify=range(42)) # Compliant
  requests.request('GET', 'https://example.domain', verify=range(2, 5)) # Compliant

  # Pathological cases for code coverage only
  requests.request('GET', 'https://example.domain', verify=thatThingIsNotACollectionConstructor(0))
  requests.request('GET', 'https://example.domain', verify=sorted([])) # FN, bool(sorted([])) == False
  ft = { 'from': 10, 'to': 100 }
  requests.request('GET', 'https://example.domain', verify=range(**ft))
  requests.request('GET', 'https://example.domain', verify=range("not numeric"))
  requests.request('GET', 'https://example.domain', verify=set(42))

def requestsSessionTest():
  import requests

  s1 = requests.Session()
  s1.verify = False # Noncompliant

  s2 = requests.Session()
  s2.verify = True

  s3 = requests.Session()
  s3.something_else = False

  s4 = requests.Session()
  s4.request("GET", "<https://expired.badssl.com>", verify=False) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                                        ^^^^^
  s4.get('https://example.domain', verify=False) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                       ^^^^^
  s4.post('https://example.domain', verify=False) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                                        ^^^^^

def urllibTests():
  # Mutably borrowed from
  # https://raw.githubusercontent.com/SonarSource/security-expected-issues/master/python/rules/vulnerabilities/\
  # RSPEC-4830%20Server%20certificates%20should%20be%20verified%20during%20SSL%E2%81%84TLS%20connections/\
  # urllib-test.py
  import urllib.request
  import ssl
  import sys

  # (S4830) - bydefault = ctx.verify_mode = ssl.CERT_NONE
  ctx1 = ssl._create_unverified_context()  # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #          ^^^^^^^^^^^^^^^^^^^^^^^^^^

  ctx2 = ssl._create_unverified_context()
  ctx2.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                      ^^^^^^^^^

  ctx3 = ssl._create_unverified_context()
  ctx3.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

  ctx4 = ssl._create_unverified_context()
  ctx4.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

  ctx5 = ssl._create_stdlib_context() # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #          ^^^^^^^^^^^^^^^^^^^^^^

  ctx6 = ssl._create_stdlib_context()
  ctx6.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                      ^^^^^^^^^

  ctx7 = ssl._create_stdlib_context()
  ctx7.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

  ctx8 = ssl._create_stdlib_context()
  ctx8.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

  ctx9 = ssl.create_default_context()  # Compliant (S4830) - bydefault = ctx.verify_mode = ssl.CERT_REQUIRED

  ctx9b = ssl.create_default_context()
  ctx9b.check_hostname = False
  ctx9b.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                       ^^^^^^^^^

  ctxA = ssl.create_default_context()
  ctxA.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

  ctxB = ssl.create_default_context()
  ctxB.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

  ctxC = ssl._create_default_https_context() # Compliant (S4830) - bydefault = ctx.verify_mode = ssl.CERT_REQUIRED

  ctxD = ssl._create_default_https_context()
  ctxD.check_hostname = False
  ctxD.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
  #                      ^^^^^^^^^

  ctxE = ssl._create_default_https_context()
  ctxE.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

  ctxF = ssl._create_default_https_context()
  ctxF.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

  # Corner cases for code coverage
  ctxC1 = ssl.there_is_no_such_symbol()
  ctxC1.verify_mode = ssl.CERT_REQUIRED

  ctxC2 = ssl._create_default_https_context()
  ctxC2.verify_mode = ssl.THAT_S_NOT_A_VALID_MODE

  ambiguous = "" if 42 * 42 < 1700 else (lambda x: x * x)
  ctxC3 = ambiguous._create_default_https_context()
  ctxC4 = notASymbol()


def multipleCtxReinitializationsWithFinalGoodSetting():
    import ssl
    ctx = ssl._create_unverified_context()  # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^

    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #                     ^^^^^^^^^

    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

def multipleCtxReinitializationsWithFinalBadSetting():
    import ssl
    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

    ctx = ssl._create_stdlib_context()
    ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #                     ^^^^^^^^^

def multipleCtxReinitializationsWithGoodSettingBeforeFinalBadDefault():
    import ssl
    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)
    ctx = ssl._create_unverified_context() # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^

def multipleCtxReinitializationsWithBadSettingBeforeFinalGoodDefault():
    import ssl
    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #                     ^^^^^^^^^

    ctx = ssl._create_default_https_context() # Compliant (S4830) - bydefault = ctx.verify_mode = ssl.CERT_REQUIRED

def ctxSymbolSharedBetweenTwoIfBranches():
    import ssl
    if ca_file is not None and hasattr(ssl, 'create_default_context'):
        ctx = ssl.create_default_context(cafile=ca_file)
        ctx.verify_mode = ssl.CERT_REQUIRED
        args['context'] = ctx

    if not verify and parse.scheme == 'https' and (
        hasattr(ssl, 'create_default_context')):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
        #                     ^^^^^^^^^

def call_expression_in_arguments():
    import ssl
    conn = httplib.client.HTTPSConnection("123.123.21.21", context=ssl._create_unverified_context()) # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
#                                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^ 0

def sslSetVerifyModeRequired():
    import ssl

    ctx1 = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)  # Noncompliant
    ctx2 = ssl.SSLContext(ssl.PROTOCOL_TLS)  # Noncompliant
    ctx3 = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)  # Compliant
    ctx4 = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # Compliant

    ctx5 = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    ctx5.verify_mode = ssl.CERT_REQUIRED  # Compliant

    ctx6 = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    ctx6.verify_mode = ssl.CERT_NONE # Noncompliant

    ctx7 = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx7.verify_mode = ssl.CERT_NONE # Noncompliant

    ctx8 = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # Compliant
    ctx8.verify_mode = ssl.CERT_OPTIONAL

def sslCheckHostNameAttributeSet():
    import ssl

    ctx1 = ssl._create_unverified_context()
    ctx1.check_hostname = True  # Compliant: this sets `verify_mode` to CERT_OPTIONAL

    ctx2 = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    ctx2.check_hostname = True  # Compliant

    ctx3 = ssl._create_stdlib_context()
    ctx3.check_hostname = True  # Compliant

    ctx4 = ssl.create_default_context()
    ctx4.check_hostname = False  # Compliant

def httpx_verify():
    import httpx

    httpx.get("https://expired.badssl.com/", verify=False)  # Noncompliant
    httpx.get("https://expired.badssl.com/")  # Compliant

    httpx.stream(method, url, verify=False) # Noncompliant
    httpx.stream(method, url)

    httpx.get(url, verify=False)    # Noncompliant
    httpx.get(url)

    httpx.options(url, verify=False)    # Noncompliant
    httpx.options(url)

    httpx.head(url, verify=False)   # Noncompliant
    httpx.head(url)

    httpx.post(url, verify=False)   # Noncompliant
    httpx.post(url)

    httpx.put(url, verify=False)    # Noncompliant
    httpx.put(url)

    httpx.patch(url, verify=False)  # Noncompliant
    httpx.patch(url)

    httpx.delete(url, verify=False) # Noncompliant
    httpx.delete(url)

    insecure_client = httpx.AsyncClient(verify=False)  # Noncompliant
    secure_client = httpx.AsyncClient(verify=True)  # Compliant

async def aiohttp_test():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        session.request('GET', "https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.request('GET', "https://expired.badssl.com/", ssl=False) # Noncompliant
        session.request('GET', "https://expired.badssl.com/", verify_ssl=False, ssl=True) # Noncompliant
        session.request('GET', "https://expired.badssl.com/", verify_ssl=True, ssl=False) # Noncompliant
        session.request('GET', "https://expired.badssl.com/") # Compliant
        session.get("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.post("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.put("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.delete("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.patch("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.head("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.options("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
        session.ws_connect("https://expired.badssl.com/", verify_ssl=False) # Noncompliant
