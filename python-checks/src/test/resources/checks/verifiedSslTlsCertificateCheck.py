def pyopensslTest():
  # Mutably borrowed from here:
  # https://github.com/SonarSource/security-expected-issues/blob/master/python/rules/vulnerabilities/\
  # RSPEC-4830%20Server%20certificates%20should%20be%20verified%20during%20SSL%E2%81%84TLS%20connections/\
  # pyopenssl-test.py

  from OpenSSL import SSL
  import sys, os, select, socket

  # Valid modes are VERIFY_NONE, VERIFY_PEER, VERIFY_CLIENT_ONCE, VERIFY_FAIL_IF_NO_PEER_CERT and defined on OpenSSL::SSL
  # The default mode is VERIFY_NONE, which does not perform any verification at al
  #
  # VERIFY_PEER Don't check peer certificate.
  # VERIFY_FAIL_IF_NO_PEER_CERT Client:Check peer certificate, Server:Check peer certificate and skip if no certificate received.
  #  If VERIFY_PEER is used, mode can be OR:ed with VERIFY_FAIL_IF_NO_PEER_CERT and VERIFY_CLIENT_ONCE to further control the behaviour.

  ctx1 = SSL.Context(SSL.TLSv1_2_METHOD)
  ctx1.set_verify(SSL.VERIFY_PEER, verify_callback) # Compliant
  ctx1.set_verify(SSL.VERIFY_PEER | SSL.VERIFY_FAIL_IF_NO_PEER_CERT, verify_callback) # Compliant
  ctx1.set_verify(SSL.VERIFY_PEER | SSL.VERIFY_FAIL_IF_NO_PEER_CERT | VERIFY_CLIENT_ONCE, verify_callback) # Compliant

  def verify_callback(connection, x509, errnum, errdepth, ok):
      if not ok:
          print("Bad Certs")
      else:
          print("Certs are fine")
      return ok

  # Initialize context
  ctx = SSL.Context(SSL.TLSv1_2_METHOD)
  ctx.set_verify(SSL.VERIFY_NONE, verify_callback) # Noncompliant {{Omitting the check of the peer certificate is dangerous.}}
  #                  ^^^^^^^^^^^

  ctx.set_verify(SSL.VERIFY_FAIL_IF_NO_PEER_CERT | SSL.VERIFY_NONE | SSL.VERIFY_CLIENT_ONCE) # Noncompliant {{Omitting the check of the peer certificate is dangerous.}}
  #                                                    ^^^^^^^^^^^

  # Set up client
  sock = SSL.Connection(ctx, socket.socket())
  sock.connect(("151.101.0.223", 443))
  sock.do_handshake()

  data="""GET / HTTP/1.1
  Host: 151.101.0.223
  """.replace("\n","\r\n")
  sock.send(data)

  while 1:
      try:
          buf = sock.recv(4096)
          print(buf)
      except SSL.Error:
          print('Connection died unexpectedly')
          break

  sock.shutdown()
  sock.close()

  # Weird cases for code coverage only.
  ctxC1 = SSL.Context(SSL.TLSv1_2_METHOD)
  ctxC1.set_verify()
  kwargs = { 'something': True }
  ctxC1.set_verify(**kwargs)
  ctxC1.set_verify(noSSL.THIS_DOESNT_EXIST)


def requestsTests():
  # Mutably borrowed from here:
  # https://github.com/SonarSource/security-expected-issues/blob/master/python/rules/vulnerabilities/\
  # RSPEC-4830%20Server%20certificates%20should%20be%20verified%20during%20SSL%E2%81%84TLS%20connections/\
  # requests-tests.py
  # Main changes: the position of the error message in `kwargs`-cases has been moved closer to kwargs, not to
  # the method invocation.
  import requests

  requests.request('GET', 'https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^^^^
  requests.request('GET', 'https://example.domain', verify='') # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=0) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^
  requests.request('GET', 'https://example.domain', verify=0.0) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^^
  requests.request('GET', 'https://example.domain', verify=0j) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify="") # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=b'') # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^^
  requests.request('GET', 'https://example.domain', verify=[]) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify={}) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=()) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^
  requests.request('GET', 'https://example.domain', verify=set()) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^^^^
  requests.request('GET', 'https://example.domain', verify=range(0)) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                        ^^^^^^^^
  requests.request(verify=False, method='GET', url='https://example.domain') # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                       ^^^^^
  kargs1 = {'verify': False} # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                   ^^^^^
  requests.request('GET', 'https://example.domain', **kargs1)
  kargs2 = {'method': 'GET', 'url': 'https://example.domain', 'verify': False} # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                                     ^^^^^
  requests.request(**kargs2)

  requests.get('https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                             ^^^^^
  requests.head('https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                              ^^^^^
  requests.post('https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                              ^^^^^
  requests.put('https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                             ^^^^^
  requests.delete('https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                                ^^^^^
  requests.patch('https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
  #                                               ^^^^^
  requests.options('https://example.domain', verify=False) # Noncompliant {{Disabling certificate verification is dangerous.}}
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

  # Pathological cases for test coverage
  requests.request('GET', 'https://example.domain', verify=thatThingIsNotACollectionConstructor(0))
  requests.request('GET', 'https://example.domain', verify=sorted([])) # FN, bool(sorted([])) == False
  ft = { 'from': 10, 'to': 100 }
  requests.request('GET', 'https://example.domain', verify=range(**ft))
  requests.request('GET', 'https://example.domain', verify=range("not numeric"))
  requests.request('GET', 'https://example.domain', verify=set(42))


def urllibTests():
  # Mutably borrowed from
  # https://raw.githubusercontent.com/SonarSource/security-expected-issues/master/python/rules/vulnerabilities/\
  # RSPEC-4830%20Server%20certificates%20should%20be%20verified%20during%20SSL%E2%81%84TLS%20connections/\
  # urllib-test.py
  import urllib.request
  import ssl
  import sys

  try:
    print("test1-default")
    # (S4830) - bydefault = ctx.verify_mode = ssl.CERT_NONE
    ctx1 = ssl._create_unverified_context()  # Noncompliant {{Certificate verification is disabled by default, verify_mode should be updated.}}
    #          ^^^^^^^^^^^^^^^^^^^^^^^^^^
    r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx1)
    print(r1.status, r1.reason)

    print("test1-1")
    ctx2 = ssl._create_unverified_context()
    ctx2.verify_mode = ssl.CERT_NONE # Noncompliant {{Disabling certificate verification is dangerous.}}
    #                      ^^^^^^^^^
    r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx2)
    print(r1.status, r1.reason)

    print("test1-2")
    ctx3 = ssl._create_unverified_context()
    ctx3.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)

    print("test1-3")
    ctx4 = ssl._create_unverified_context()
    ctx4.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)


    print("test2-default")
    # (S4830) - bydefault = ctx.verify_mode = ssl.CERT_NONE
    ctx5 = ssl._create_stdlib_context() # Noncompliant {{Certificate verification is disabled by default, verify_mode should be updated.}}
    #          ^^^^^^^^^^^^^^^^^^^^^^
    r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx5)
    print(r1.status, r1.reason)

    print("test2-1")
    ctx6 = ssl._create_stdlib_context()
    ctx6.verify_mode = ssl.CERT_NONE # Noncompliant {{Disabling certificate verification is dangerous.}}
    #                      ^^^^^^^^^
    r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx6)
    print(r1.status, r1.reason)

    print("test2-2")
    ctx7 = ssl._create_stdlib_context()
    ctx7.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)

    print("test3-3")
    ctx8 = ssl._create_stdlib_context()
    ctx8.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)


    print("test3-default")
    ctx9 = ssl.create_default_context()  # Compliant (S4830) - bydefault = ctx.verify_mode = ssl.CERT_REQUIRED
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)

    print("test3-1")
    ctx9b = ssl.create_default_context()
    ctx9b.check_hostname = False
    ctx9b.verify_mode = ssl.CERT_NONE # Noncompliant {{Disabling certificate verification is dangerous.}}
    #                       ^^^^^^^^^
    r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx9b)
    print(r1.status, r1.reason)

    print("test3-2")
    ctxA = ssl.create_default_context()
    ctxA.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)

    print("test3-3")
    ctxB = ssl.create_default_context()
    ctxB.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)


    print("test4-default")
    ctxC = ssl._create_default_https_context() # Compliant (S4830) - bydefault = ctx.verify_mode = ssl.CERT_REQUIRED
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)

    print("test4-1")
    ctxD = ssl._create_default_https_context()
    ctxD.check_hostname = False
    ctxD.verify_mode = ssl.CERT_NONE # Noncompliant {{Disabling certificate verification is dangerous.}}
    #                      ^^^^^^^^^
    r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctxD)
    print(r1.status, r1.reason)

    print("test4-2")
    ctxE = ssl._create_default_https_context()
    ctxE.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)

    print("test4-3")
    ctxF = ssl._create_default_https_context()
    ctxF.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)
    #r1 = urllib.request.urlopen('https://self-signed.badssl.com/', context=ctx)
    print(r1.status, r1.reason)

    # Corner cases for code coverage
    ctxC1 = ssl.there_is_no_such_symbol()
    ctxC1.verify_mode = ssl.CERT_REQUIRED

    ctxC2 = ssl._create_default_https_context()
    ctxC2.verify_mode = ssl.THAT_S_NOT_A_VALID_MODE

    ambiguous = "" if 42 * 42 < 1700 else (lambda x: x * x)
    ctxC3 = ambiguous._create_default_https_context()
    ctxC4 = notASymbol()


  except: # catch *all* exceptions
    e = sys.exc_info()[0]
    print( "<p>Error: %s</p>" % e )
    raise
