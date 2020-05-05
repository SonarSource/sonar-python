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
