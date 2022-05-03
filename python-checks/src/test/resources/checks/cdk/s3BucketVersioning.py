from aws_cdk import aws_s3 as s3

bucket = s3.Bucket(self, "MyUnversionedBucket", versioned=False)  # Noncompliant
#                                               ^^^^^^^^^^^^^^^

# Noncompliant@+1 {{Omitting the "versioned" argument disables S3 bucket versioning. Make sure it is safe here.}}
bucket = s3.Bucket(self, "MyUnversionedBucket")
#        ^^^^^^^^^

bucket_versioning = False  # Noncompliant
bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=bucket_versioning  # Noncompliant {{Make sure an unversioned S3 bucket is safe here.}}
                #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   )

bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=True  # Compliant
                   )

bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=unresolved_var  # Compliant
                   )

bucket_versioning1 = False  # Noncompliant {{Propagated setting}}
second_versioning = bucket_versioning1
bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=second_versioning  # Noncompliant {{Make sure an unversioned S3 bucket is safe here.}}
                #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   )

third_versioning = True
bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=third_versioning  # Compliant
                   )

bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=True  # Compliant
                   )

value = "FALSE"
bucket = s3.Bucket(self, "MyUnversionedBucket",
                   versioned=value # Compliant
                   )

a = function_call_for_coverage(10)
