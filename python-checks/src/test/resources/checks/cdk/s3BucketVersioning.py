from aws_cdk import aws_s3 as s3

a = int(10)

bucket = s3.Bucket(self, "MyUnversionedBucket", versioned=False) # Noncompliant
                                              # ^^^^^^^^^^^^^^^

# Noncompliant@+1 {{Make sure using unversioned S3 bucket is safe here. Omitting 'versioned=True' disables S3 bucket versioning. Make sure it is safe here.}}
bucket = s3.Bucket(self, "MyUnversionedBucket")
#        ^^^^^^^^^

bucket_versioning = False
bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=bucket_versioning # Noncompliant
                   # highlight `versioned=bucket_versioning` as 2st location
                   )

bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=True      # Compliant
                   )

bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=unresolved_var      # Compliant
                   )

second_versioning = bucket_versioning
bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=second_versioning # Noncompliant
                   )

third_versioning = True
bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=third_versioning # Compliant
                   )
