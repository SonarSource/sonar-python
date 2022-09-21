from aws_cdk.aws_opensearchservice import Domain, EncryptionAtRestOptions, EngineVersion

class DomainStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        encryptionRestOptionMethodGood = EncryptionAtRestOptions(enabled=True)
        encryptionRestOptionMethodBad = EncryptionAtRestOptions(enabled=False)
        encryptionRestOptionDictionaryGood = {"enabled": True}
        encryptionRestOptionDictionaryBad = {"enabled": False}
        encrypted = True
        not_encrypted = False

        Domain(self, "unencrypted-domain-1", version=EngineVersion.OPENSEARCH_1_3) # NonCompliant{{Omitting encryption_at_rest causes encryption of data at rest to be disabled for this {OpenSearch|Elasticsearch} domain. Make sure it is safe here.}}
        Domain(self, "unencrypted-domain-2", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=EncryptionAtRestOptions) # NonCompliant{{Make sure that using unencrypted {OpenSearch|Elasticsearch} domains is safe here.}}
        Domain(self, "unencrypted-domain-3", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=EncryptionAtRestOptions()) # NonCompliant
        Domain(self, "unencrypted-domain-4", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=EncryptionAtRestOptions(enabled=False)) # NonCompliant{{Make sure that using unencrypted {OpenSearch|Elasticsearch} domains is safe here.}}
        Domain(self, "unencrypted-domain-5", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionMethodBad) # NonCompliant
        Domain(self, "unencrypted-domain-6", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": False}) # NonCompliant
        Domain(self, "unencrypted-domain-6", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"another_key": False}) # NonCompliant
        Domain(self, "unencrypted-domain-7", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionDictionaryBad) # NonCompliant
        Domain(self, "unencrypted-domain-8", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=EncryptionAtRestOptions(enabled=not_encrypted)) # NonCompliant
        Domain(self, "unencrypted-domain-9", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": not_encrypted}) # NonCompliant
        Domain(self, "encrypted-domain-1", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=EncryptionAtRestOptions(enabled=True))
        Domain(self, "encrypted-domain-2", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionMethodGood)
        Domain(self, "encrypted-domain-3", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": True})
        Domain(self, "encrypted-domain-4", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionDictionaryGood)
        Domain(self, "encrypted-domain-5", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest=EncryptionAtRestOptions(enabled=encrypted))
        Domain(self, "encrypted-domain-6", version=EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": encrypted})

from aws_cdk.aws_opensearchservice import CfnDomain

class CfnDomainStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        encrypted = True
        not_encrypted = False

        CfnDomain(self, "unencrypted-domain-1") # NonCompliant{{Omitting encryption_at_rest_options causes encryption of data at rest to be disabled for this {OpenSearch|Elasticsearch} domain. Make sure it is safe here.}}
        CfnDomain(self, "unencrypted-domain-2", encryption_at_rest_options=CfnDomain.EncryptionAtRestOptionsProperty()) # NonCompliant{{Make sure that using unencrypted {OpenSearch|Elasticsearch} domains is safe here.}}
        CfnDomain(self, "unencrypted-domain-3", encryption_at_rest_options=CfnDomain.EncryptionAtRestOptionsProperty) # NonCompliant
        CfnDomain(self, "unencrypted-domain-4", encryption_at_rest_options=CfnDomain.EncryptionAtRestOptionsProperty(enabled=False)) # NonCompliant{{Make sure that using unencrypted {OpenSearch|Elasticsearch} domains is safe here.}}
        CfnDomain(self, "unencrypted-domain-5", encryption_at_rest_options={"enabled": False}) # NonCompliant
        CfnDomain(self, "unencrypted-domain-6", encryption_at_rest_options=CfnDomain.EncryptionAtRestOptionsProperty(enabled=not_encrypted)) # NonCompliant
        CfnDomain(self, "unencrypted-domain-7", encryption_at_rest_options={"enabled": not_encrypted}) # NonCompliant
        CfnDomain(self, "encrypted-domain-1", encryption_at_rest_options=CfnDomain.EncryptionAtRestOptionsProperty(enabled=True))
        CfnDomain(self, "encrypted-domain-2", encryption_at_rest_options={"enabled": True})
        CfnDomain(self, "encrypted-domain-3", encryption_at_rest_options=CfnDomain.EncryptionAtRestOptionsProperty(enabled=encrypted))
        CfnDomain(self, "encrypted-domain-4", encryption_at_rest_options={"enabled": encrypted})
        CfnDomain(self, "encrypted-domain-2", encryption_at_rest_options={1:1, "enabled": True}) # code coverage on !StringLiteral key
