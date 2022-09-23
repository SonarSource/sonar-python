from aws_cdk import (aws_sns as sns)

noneKey = None
validKey = kms.Key(self, "key")

sns.Topic(self, "unencrypted") # NonCompliant{{Omitting "master_key" disables SNS topics encryption. Make sure it is safe here.}}
sns.Topic(self, "unencrypted", master_key=noneKey) # NonCompliant{{Omitting "master_key" disables SNS topics encryption. Make sure it is safe here.}}
sns.Topic(self, "encrypted", master_key=kms.Key(self, "key")) # Compliant
sns.Topic(self, "encrypted", master_key=validKey) # Compliant

sns.CfnTopic(self, "unencrypted") # NonCompliant{{Omitting "kms_master_key_id" disables SNS topics encryption. Make sure it is safe here.}}
sns.CfnTopic(self, "unencrypted", kms_master_key_id=noneKey) # NonCompliant{{Omitting "kms_master_key_id" disables SNS topics encryption. Make sure it is safe here.}}
sns.CfnTopic(self, "encrypted-selfmanaged", kms_master_key_id=kms.Key(self, "key")) # Compliant
sns.CfnTopic(self, "encrypted-selfmanaged", kms_master_key_id=someKey) # Compliant
