def pyopensslTest():
  # Mutably borrowed from here:
  # https://github.com/SonarSource/security-expected-issues/blob/master/python/rules/vulnerabilities/\
  # RSPEC-4830%20Server%20certificates%20should%20be%20verified%20during%20SSL%E2%81%84TLS%20connections/pyopenssl-test.py

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
