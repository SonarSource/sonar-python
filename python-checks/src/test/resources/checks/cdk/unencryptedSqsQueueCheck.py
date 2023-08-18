from aws_cdk import (aws_sqs as sqs)

# Success
customKey = my_key.key_id
enabled_sqs = True
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=my_key.key_id)
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=customKey)
sqs.CfnQueue(self, "encrypted", sqs_managed_sse_enabled=True)
sqs.CfnQueue(self, "encrypted", sqs_managed_sse_enabled=enabled_sqs)

sqs.CfnQueue(self, "unencrypted")  # Compliant ref: SONARPY-1416

encryptionParam = sqs.QueueEncryption.KMS
sqs.Queue(self, "encrypted-managed", encryption=sqs.QueueEncryption.KMS_MANAGED)
sqs.Queue(self, "encrypted-managed", encryption=sqs.QueueEncryption.KMS)
sqs.Queue(self, "encrypted-managed", encryption=encryptionParam)

encryptionNone = None
sqs.Queue(self, "unencrypted-explicit")  # Compliant ref: SONARPY-1416
sqs.Queue(self, "unencrypted-explicit", encryption=sqs.QueueEncryption.UNENCRYPTED) # Compliant ref: SONARPY-1416
sqs.Queue(self, "unencrypted-explicit", encryption=None) # Compliant ref: SONARPY-1416  
sqs.Queue(self, "unencrypted-explicit", encryption=encryptionNone) # Compliant ref: SONARPY-1416

# Failing cases
noneKey = None
not_enabled_sqs = False
sqs.CfnQueue(self, "unencrypted", sqs_managed_sse_enabled=False) # NonCompliant{{Setting "sqs_managed_sse_enabled" to "false" disables SQS queues encryption. Make sure it is safe here.}}
sqs.CfnQueue(self, "unencrypted", sqs_managed_sse_enabled=not_enabled_sqs) # NonCompliant{{Setting "sqs_managed_sse_enabled" to "false" disables SQS queues encryption. Make sure it is safe here.}}
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=None) # NonCompliant{{Setting "kms_master_key_id" to "None" disables SQS queues encryption. Make sure it is safe here.}}
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=noneKey) # NonCompliant{{Setting "kms_master_key_id" to "None" disables SQS queues encryption. Make sure it is safe here.}}

