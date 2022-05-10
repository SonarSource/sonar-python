def grant_compliant():
    foo = Foo()
    foo.grant_public_access() # Compliant since no aws_cdk import
