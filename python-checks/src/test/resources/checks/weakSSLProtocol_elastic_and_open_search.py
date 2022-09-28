# OpenSearch & ElasticSearch have identical use, they just have different import, but method and parameter for tls_security_policy are the same
## Domain - default tls_security_policy : TLSSecurityPolicy.TLS_1_0
from aws_cdk import (aws_opensearchservice as opensearch)
from aws_cdk import (aws_elasticsearch as elasticsearch)

class OpenSearchStack(Stack):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # a normal call to Domain look like this, for the sake of simplicity/readability, we will remove the arguments not checked by the rules
        opensearch.Domain(self, "default", version=opensearch.EngineVersion.OPENSEARCH_1_0, tls_security_policy=opensearch.TLSSecurityPolicy.TLS_1_2)

        # Variables
        os_tls10 = opensearch.TLSSecurityPolicy.TLS_1_0
        os_tls12 = opensearch.TLSSecurityPolicy.TLS_1_2
        es_tls10 = elasticsearch.TLSSecurityPolicy.TLS_1_0
        es_tls12 = elasticsearch.TLSSecurityPolicy.TLS_1_2

        # Sensitive test case
        opensearch.Domain() # Noncompliant
        opensearch.Domain(tls_security_policy=opensearch.TLSSecurityPolicy.TLS_1_0) # Noncompliant
        opensearch.Domain(tls_security_policy=os_tls10) # Noncompliant
        elasticsearch.Domain() # Noncompliant
        elasticsearch.Domain(tls_security_policy=elasticsearch.TLSSecurityPolicy.TLS_1_0) # Noncompliant
        elasticsearch.Domain(tls_security_policy=es_tls10) # Noncompliant

        # Compliant test case
        elasticsearch.Domain(tls_security_policy=elasticsearch.TLSSecurityPolicy.TLS_1_2) # Compliant
        elasticsearch.Domain(tls_security_policy=es_tls12) # Compliant


## CfnDomain - default tls_security_policy : Policy-Min-TLS-1-0-2019-07
class CfnOpenSearchStack(Stack):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # a normal call to CfnDomain look like this, for the sake of simplicity/readability, we will remove the arguments not checked by the rules
        opensearch.CfnDomain(self, "compliant", domain_endpoint_options=opensearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy="Policy-Min-TLS-1-2-2019-07"))

        # Variables
        str_tls_10 = "Policy-Min-TLS-1-0-2019-07"
        str_tls_12 = "Policy-Min-TLS-1-2-2019-07"
        domain_endpoint_options_tls_10 = opensearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy="Policy-Min-TLS-1-0-2019-07")
        domain_endpoint_options_tls_12 = opensearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy="Policy-Min-TLS-1-2-2019-07")
        dict_tls_10 = {"tls_security_policy":"Policy-Min-TLS-1-0-2019-07"}
        dict_tls_12 = {"tls_security_policy":"Policy-Min-TLS-1-2-2019-07"}

        # Sensitive test case
        opensearch.CfnDomain() # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options=opensearch.CfnDomain.DomainEndpointOptionsProperty) # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options=opensearch.CfnDomain.DomainEndpointOptionsProperty()) # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options=opensearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy="Policy-Min-TLS-1-0-2019-07")) # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options=opensearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy=str_tls_10)) # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options={"any_key":"any_value"}) # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options={"tls_security_policy":"Policy-Min-TLS-1-0-2019-07"}) # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options={"tls_security_policy":str_tls_10}) # Noncompliant
        opensearch.CfnDomain(domain_endpoint_options=dict_tls_10) # Noncompliant
        elasticsearch.CfnDomain(domain_endpoint_options=elasticsearch.CfnDomain.DomainEndpointOptionsProperty()) # Noncompliant

        # Compliant test case
        opensearch.CfnDomain(domain_endpoint_options=opensearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy="Policy-Min-TLS-1-2-2019-07"))
        opensearch.CfnDomain(domain_endpoint_options=opensearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy=str_tls_12))
        opensearch.CfnDomain(domain_endpoint_options={"tls_security_policy":"Policy-Min-TLS-1-2-2019-07"})
        opensearch.CfnDomain(domain_endpoint_options={"any_key":"any_value", "tls_security_policy":"Policy-Min-TLS-1-2-2019-07"})
        opensearch.CfnDomain(domain_endpoint_options={"tls_security_policy":str_tls_12})
        opensearch.CfnDomain(domain_endpoint_options=[])
        opensearch.CfnDomain(domain_endpoint_options=dict_tls_12)
        elasticsearch.CfnDomain(domain_endpoint_options=elasticsearch.CfnDomain.DomainEndpointOptionsProperty(tls_security_policy="Policy-Min-TLS-1-2-2019-07"))
