def fixupRequestBulletPoint3WallOfFalseNegatives():
    import ssl
    ctx = ssl._create_unverified_context()  # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^

    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #                     ^^^^^^^^^

    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

    ctx = ssl._create_unverified_context()
    ctx.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

    ctx = ssl._create_stdlib_context()  # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #         ^^^^^^^^^^^^^^^^^^^^^^

    ctx = ssl._create_stdlib_context()
    ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #                     ^^^^^^^^^

    ctx = ssl._create_stdlib_context()
    ctx.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

    ctx = ssl._create_stdlib_context()
    ctx.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

    ctx = ssl.create_default_context()  # Compliant (S4830) - bydefault = ctx.verify_mode = ssl.CERT_REQUIRED

    ctx = ssl.create_default_context()
    ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #                     ^^^^^^^^^

    ctx = ssl.create_default_context()
    ctx.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

    ctx = ssl.create_default_context()
    ctx.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)

    ctx = ssl._create_default_https_context() # Compliant (S4830) - bydefault = ctx.verify_mode = ssl.CERT_REQUIRED

    ctx = ssl._create_default_https_context()
    ctx.verify_mode = ssl.CERT_NONE # Noncompliant {{Enable server certificate validation on this SSL/TLS connection.}}
    #                     ^^^^^^^^^

    ctx = ssl._create_default_https_context()
    ctx.verify_mode = ssl.CERT_OPTIONAL # Compliant (S4830)

    ctx = ssl._create_default_https_context()
    ctx.verify_mode = ssl.CERT_REQUIRED # Compliant (S4830)
