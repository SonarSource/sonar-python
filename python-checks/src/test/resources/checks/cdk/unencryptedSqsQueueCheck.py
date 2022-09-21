from aws_cdk import (aws_sqs as sqs)

# Success
customKey = my_key.key_id
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=my_key.key_id)
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=customKey)

encryptionParam = sqs.QueueEncryption.KMS
sqs.Queue(self, "encrypted-managed", encryption=sqs.QueueEncryption.KMS_MANAGED)
sqs.Queue(self, "encrypted-managed", encryption=sqs.QueueEncryption.KMS)
sqs.Queue(self, "encrypted-managed", encryption=encryptionParam)

# Failing cases
noneKey = None
sqs.CfnQueue(self, "unencrypted") # NonCompliant{{Omitting "kms_master_key_id" disables SQS queues encryption. Make sure it is safe here.}}
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=None) # NonCompliant{{Setting "kms_master_key_id" to "None" disables SQS queues encryption. Make sure it is safe here.}}
sqs.CfnQueue(self, "encrypted-selfmanaged", kms_master_key_id=noneKey) # NonCompliant{{Setting "kms_master_key_id" to "None" disables SQS queues encryption. Make sure it is safe here.}}

encryptionNone = None
sqs.Queue(self, "unencrypted-explicit") # NonCompliant {{Omitting "encryption" disables SQS queues encryption. Make sure it is safe here.}}
sqs.Queue(self, "unencrypted-explicit", encryption=sqs.QueueEncryption.UNENCRYPTED) # NonCompliant {{Setting "encryption" to "QueueEncryption.UNENCRYPTED" disables SQS queues encryption. Make sure it is safe here.}}
sqs.Queue(self, "unencrypted-explicit", encryption=None) # NonCompliant {{Setting "encryption" to "None" disables SQS queues encryption. Make sure it is safe here.}}
sqs.Queue(self, "unencrypted-explicit", encryption=encryptionNone) # NonCompliant {{Setting "encryption" to "None" disables SQS queues encryption. Make sure it is safe here.}}
