from aws_cdk import (aws_apigateway as apigateway, aws_apigatewayv2 as apigatewayv2)

class ExampleStack():
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # a normal call to DomainName look like this, for the sake of simplicity/readability, we will remove the arguments not checked by the rules
        apigateway.DomainName(self, "example", domain_name="example.com", certificate=certificate, security_policy=apigateway.SecurityPolicy.TLS_1_2)

        # Variables
        tls10 = apigateway.SecurityPolicy.TLS_1_0
        tls12 = apigateway.SecurityPolicy.TLS_1_2
        tls10v2 = apigatewayv2.SecurityPolicy.TLS_1_0
        tls12v2 = apigatewayv2.SecurityPolicy.TLS_1_2

        # Sensitive test case
        apigateway.DomainName(security_policy=apigateway.SecurityPolicy.TLS_1_0) # Noncompliant
        apigateway.DomainName(security_policy=tls10) # Noncompliant
        apigatewayv2.DomainName(security_policy=apigatewayv2.SecurityPolicy.TLS_1_0) # Noncompliant
        apigatewayv2.DomainName(security_policy=tls10v2) # Noncompliant

        # Compliant test case
        apigateway.DomainName()
        apigateway.DomainName(security_policy=apigateway.SecurityPolicy.TLS_1_2)
        apigateway.DomainName(security_policy=tls12)
        apigatewayv2.DomainName()
        apigatewayv2.DomainName(security_policy=apigatewayv2.SecurityPolicy.TLS_1_2)
        apigatewayv2.DomainName(security_policy=tls12v2)

class ExampleStack(Stack):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # a normal call to CfnDomain look like this, for the sake of simplicity/readability, we will remove the arguments not checked by the rules
        apigateway.CfnDomainName(self, "compliant", domain_name="compliant.example.com", security_policy="TLS_1_2")

        # Variables
        tls10 = "TLS_1_0"
        tls12 = "TLS_1_2"

        # Sensitive test case
        apigateway.CfnDomainName(security_policy="TLS_1_0") # Noncompliant
        apigateway.CfnDomainName(security_policy=tls10) # Noncompliant

        # Compliant test case
        apigateway.CfnDomainName()
        apigateway.CfnDomainName(security_policy="TLS_1_2")
        apigateway.CfnDomainName(security_policy=tls12)
        apigateway.CfnDomainName(security_policy="TLS_1_1") # Could be sensitive but it is not a valid value for CfnDomainName

