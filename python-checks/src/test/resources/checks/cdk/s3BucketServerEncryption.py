from aws_cdk import aws_s3 as s3

bucket = s3.Bucket(self,"MyUnencryptedBucket", encryption=s3.BucketEncryption.UNENCRYPTED) # NonCompliant {{Omitting 'encryption' and 'encryption_key' disables server-side encryption. Make sure it is safe here.}}
#        ^^^^^^^^^

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
bucket = s3.Bucket(self,"MyUnencryptedBucket",  # NonCompliant
                   encryption=s3.BucketEncryption.UNENCRYPTED,
                   encryption_key=my_encryption_key
)
bucket = s3.Bucket(self,"MyUnencryptedBucket") # NonCompliant

type_KMS = s3.BucketEncryption.KMS
bucket = s3.Bucket(self,"MyEncryptedBucket",
                   encryption=type_KMS          # Compliant
                   )

bucket = s3.Bucket(self,"MyEncryptedBucket",    # NonCompliant
                   encryption=s3.BucketEncryption.S3_MANAGED,
                    encryption_key=my_encryption_key)
