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

        aws_os.Domain(self, "unencrypted-domain-1", version=aws_os.EngineVersion.OPENSEARCH_1_3) # NonCompliant{{Omitting encryption_at_rest causes encryption of data at rest to be disabled for this OpenSearch domain. Make sure it is safe here.}}
        aws_os.Domain(self, "unencrypted-domain-2", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=aws_os.EncryptionAtRestOptions) # NonCompliant{{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.Domain(self, "unencrypted-domain-3", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=aws_os.EncryptionAtRestOptions()) # NonCompliant
        aws_os.Domain(self, "unencrypted-domain-4", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=False)) # NonCompliant{{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.Domain(self, "unencrypted-domain-5", version=aws_os.aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionMethodBad) # NonCompliant
        aws_os.Domain(self, "unencrypted-domain-6", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": False}) # NonCompliant
        aws_os.Domain(self, "unencrypted-domain-6", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest={}) # NonCompliant
        aws_os.Domain(self, "unencrypted-domain-6", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"another_key": False}) # NonCompliant
        aws_os.Domain(self, "unencrypted-domain-7", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionDictionaryBad) # NonCompliant
        aws_os.Domain(self, "unencrypted-domain-8", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=not_encrypted)) # NonCompliant
        aws_os.Domain(self, "unencrypted-domain-9", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": not_encrypted}) # NonCompliant
        aws_os.Domain(self, "encrypted-domain-1", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=True))
        aws_os.Domain(self, "encrypted-domain-2", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionMethodGood)
        aws_os.Domain(self, "encrypted-domain-3", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": True})
        aws_os.Domain(self, "encrypted-domain-4", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=encryptionRestOptionDictionaryGood)
        aws_os.Domain(self, "encrypted-domain-5", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest=aws_os.EncryptionAtRestOptions(enabled=encrypted))
        aws_os.Domain(self, "encrypted-domain-6", version=aws_os.EngineVersion.OPENSEARCH_1_3, encryption_at_rest={"enabled": encrypted})

        aws_es.Domain(self, "unencrypted-domain-1", version=aws_es.ElasticsearchVersion.V7_4) # NonCompliant{{Omitting encryption_at_rest causes encryption of data at rest to be disabled for this Elasticsearch domain. Make sure it is safe here.}}
        aws_es.Domain(self, "unencrypted-domain-2", version=aws_es.ElasticsearchVersion.V7_4, encryption_at_rest=aws_es.EncryptionAtRestOptions) # NonCompliant{{Make sure that using unencrypted Elasticsearch domains is safe here.}}

class CfnDomainStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        encrypted = True
        not_encrypted = False

        aws_os.CfnDomain(self, "unencrypted-domain-1") # NonCompliant{{Omitting encryption_at_rest_options causes encryption of data at rest to be disabled for this OpenSearch domain. Make sure it is safe here.}}
        aws_os.CfnDomain(self, "unencrypted-domain-2", encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty()) # NonCompliant{{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.CfnDomain(self, "unencrypted-domain-3", encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty) # NonCompliant
        aws_os.CfnDomain(self, "unencrypted-domain-4", encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=False)) # NonCompliant{{Make sure that using unencrypted OpenSearch domains is safe here.}}
        aws_os.CfnDomain(self, "unencrypted-domain-5", encryption_at_rest_options={"enabled": False}) # NonCompliant
        aws_os.CfnDomain(self, "unencrypted-domain-5", encryption_at_rest_options={}) # NonCompliant
        aws_os.CfnDomain(self, "unencrypted-domain-5", encryption_at_rest_options={"another_key": False}) # NonCompliant
        aws_os.CfnDomain(self, "unencrypted-domain-6", encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=not_encrypted)) # NonCompliant
        aws_os.CfnDomain(self, "unencrypted-domain-7", encryption_at_rest_options={"enabled": not_encrypted}) # NonCompliant
        aws_os.CfnDomain(self, "encrypted-domain-1", encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=True))
        aws_os.CfnDomain(self, "encrypted-domain-2", encryption_at_rest_options={"enabled": True})
        aws_os.CfnDomain(self, "encrypted-domain-3", encryption_at_rest_options=aws_os.CfnDomain.EncryptionAtRestOptionsProperty(enabled=encrypted))
        aws_os.CfnDomain(self, "encrypted-domain-4", encryption_at_rest_options={"enabled": encrypted})
        aws_os.CfnDomain(self, "encrypted-domain-2", encryption_at_rest_options={1:1, "enabled": True}) # code coverage on !StringLiteral key

        aws_es.CfnDomain(self, "unencrypted-domain-1") # NonCompliant{{Omitting encryption_at_rest_options causes encryption of data at rest to be disabled for this Elasticsearch domain. Make sure it is safe here.}}
        aws_es.CfnDomain(self, "unencrypted-domain-2", encryption_at_rest_options=aws_es.CfnDomain.EncryptionAtRestOptionsProperty()) # NonCompliant{{Make sure that using unencrypted Elasticsearch domains is safe here.}}
