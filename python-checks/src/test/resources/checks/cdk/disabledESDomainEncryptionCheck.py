import aws_cdk.aws_opensearchservice as aws_os
import aws_cdk.aws_elasticsearch as aws_es

class DomainStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        encryptionRestOptionMethodGood = aws_os.EncryptionAtRestOptions(enabled=True)
        encryptionRestOptionMethodBad = aws_os.EncryptionAtRestOptions(enabled=False)
        encryptionRestOptionDictionaryGood = {"enabled": True}
        encryptionRestOptionDictionaryBad = {"enabled": False}
        encrypted = True
        not_encrypted = False

        # A normal call to Domain has below arguments. For the sake of simplicity/readability, we will remove the ones non-related to the rule.
        aws_os.Domain(self, "unencrypted-domain", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": True})

        # Sensitive test cases
        aws_os.Domain() # NonCompliant{{Omitting encryption_at_rest causes encryption of data at rest to be disabled for this OpenSearch domain. Make sure it is safe here.}}
        aws_os.Domain(encryption_at_rest=aws_os.EncryptionAtRestOptions)
        aws_os.Domain(encryption_at_rest=aws_os.EncryptionAtRestOptions()) # NonCompliant {{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.Domain(encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=False)) # NonCompliant{{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.Domain(encryption_at_rest=encryptionRestOptionMethodBad) # NonCompliant
        aws_os.Domain(encryption_at_rest={"enabled": False}) # NonCompliant
        aws_os.Domain(encryption_at_rest={}) # NonCompliant
        aws_os.Domain(encryption_at_rest={"another_key": False}) # NonCompliant

        aws_os.Domain(encryption_at_rest=encryptionRestOptionDictionaryBad) # NonCompliant
        aws_os.Domain(encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=not_encrypted)) # NonCompliant
        aws_os.Domain(encryption_at_rest={"enabled": not_encrypted}) # NonCompliant
        aws_es.Domain() # NonCompliant{{Omitting encryption_at_rest causes encryption of data at rest to be disabled for this Elasticsearch domain. Make sure it is safe here.}}
        aws_es.Domain(encryption_at_rest=aws_es.EncryptionAtRestOptions)

        # Compliant test cases
        aws_os.Domain(encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=True))
        aws_os.Domain(encryption_at_rest=encryptionRestOptionMethodGood)
        aws_os.Domain(encryption_at_rest={"enabled": True})
        aws_os.Domain(encryption_at_rest=encryptionRestOptionDictionaryGood)
        aws_os.Domain(encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=encrypted))
        aws_os.Domain(encryption_at_rest={"enabled": encrypted})


class CfnDomainStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        encrypted = True
        not_encrypted = False
        encryptionRestOptionPropertyMethodGood = aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=True)
        encryptionRestOptionPropertyMethodBad = aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=False)
        encryptionRestOptionPropertyDictionaryGood = {"enabled": True}
        encryptionRestOptionPropertyDictionaryBad = {"enabled": False}

        # A normal call to CfnDomain has below arguments. For the sake of simplicity/readability, we will remove the ones non-related to the rule.
        aws_os.CfnDomain(self, "encrypted-domain", encryption_at_rest_options={"enabled": True})

        # Sensitive test cases
        aws_os.CfnDomain() # NonCompliant{{Omitting encryption_at_rest_options causes encryption of data at rest to be disabled for this OpenSearch domain. Make sure it is safe here.}}
        aws_os.CfnDomain(encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty()) # NonCompliant{{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.CfnDomain(encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty)
        aws_os.CfnDomain(encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=False)) # NonCompliant{{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.CfnDomain(encryption_at_rest_options=encryptionRestOptionPropertyMethodBad) # NonCompliant
        aws_os.CfnDomain(encryption_at_rest_options={"enabled": False}) # NonCompliant
        aws_os.CfnDomain(encryption_at_rest_options=encryptionRestOptionPropertyDictionaryBad) # NonCompliant
        aws_os.CfnDomain(encryption_at_rest_options={}) # NonCompliant
        aws_os.CfnDomain(encryption_at_rest_options={"another_key": False}) # NonCompliant
        aws_os.CfnDomain(encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=not_encrypted)) # NonCompliant
        aws_os.CfnDomain(encryption_at_rest_options={"enabled": not_encrypted}) # NonCompliant

        aws_es.CfnDomain() # NonCompliant{{Omitting encryption_at_rest_options causes encryption of data at rest to be disabled for this Elasticsearch domain. Make sure it is safe here.}}
        aws_es.CfnDomain(encryption_at_rest_options=aws_es.CfnDomain.EncryptionAtRestOptionsProperty()) # NonCompliant{{Make sure that using unencrypted Elasticsearch domains is safe here.}}

        # Compliant test cases
        aws_os.CfnDomain(encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=True))
        aws_os.CfnDomain(encryption_at_rest_options=encryptionRestOptionPropertyMethodGood)
        aws_os.CfnDomain(encryption_at_rest_options={"enabled": True})
        aws_os.CfnDomain(encryption_at_rest_options=encryptionRestOptionPropertyDictionaryGood)
        aws_os.CfnDomain(encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=encrypted))
        aws_os.CfnDomain(encryption_at_rest_options={"enabled": encrypted})
        aws_os.CfnDomain(encryption_at_rest_options={1:1, "enabled": True}) # code coverage on !StringLiteral key
