from aws_cdk import aws_s3 as s3

bucket = s3.Bucket(self,"MyUnencryptedBucket") # NonCompliant {{Omitting 'encryption' disables server-side encryption. Make sure it is safe here.}}
#        ^^^^^^^^^

bucket = s3.Bucket(self,"MyUnencryptedBucket", encryption=s3.BucketEncryption.UNENCRYPTED) # NonCompliant {{Objects in the bucket are not encrypted. Make sure it is safe here.}}
#                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

bucket = s3.Bucket(self,"MyEncryptedBucket",
                   encryption=s3.BucketEncryption.KMS_MANAGED      # Compliant
                   )
bucket = s3.Bucket(self,"MyEncryptedBucket",
                   encryption=s3.BucketEncryption.S3_MANAGED          # Compliant
                   )
bucket = s3.Bucket(self,"MyEncryptedBucket",
                   encryption=s3.BucketEncryption.KMS          # Compliant
                   )
bucket = s3.Bucket(self,"MyEncryptedBucket",
                   encryption_key=my_encryption_key                 # Compliant
                   )

bucket = s3.Bucket(self,"MyUnencryptedBucket",
                   encryption=s3.BucketEncryption.UNENCRYPTED, # NonCompliant {{Objects in the bucket are not encrypted. Make sure it is safe here.}}
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   encryption_key=my_encryption_key
)

type_KMS = s3.BucketEncryption.KMS
bucket = s3.Bucket(self,"MyEncryptedBucket",
                   encryption=type_KMS          # Compliant
                   )

my_encryption_key2 = aws_cdk.aws_kms.IKey() # Compliant
bucket = s3.Bucket(self,"MyEncryptedBucket",
                   encryption=s3.BucketEncryption.S3_MANAGED,
                    encryption_key=my_encryption_key2)

def create_bucket():
    type_unencrypted = s3.BucketEncryption.UNENCRYPTED
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
    bucket = s3.Bucket(self,"MyEncryptedBucket",
                       encryption=type_unencrypted,     # NonCompliant {{Objects in the bucket are not encrypted. Make sure it is safe here.}}
#                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       encryption_key=my_encryption_key2)


coverage = s3.Bucket(self, "bucket", encryption=b)  # Compliant
