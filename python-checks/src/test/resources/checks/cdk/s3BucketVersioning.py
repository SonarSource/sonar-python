from aws_cdk import aws_s3 as s3

# Noncompliant@+1 {{Omitting the "versioned" argument disables S3 bucket versioning. Make sure it is safe here.}}
bucket = s3.Bucket(self, "MyUnversionedBucket")
#        ^^^^^^^^^

bucket = s3.Bucket(self, "MyUnversionedBucket", versioned=False)  # Noncompliant {{Make sure an unversioned S3 bucket is safe here.}}
#                                               ^^^^^^^^^^^^^^^

bucket = s3.Bucket(self, "MyUnversionedBucket", versioned=True)  # Compliant

bucket = s3.Bucket(self, "MyUnversionedBucket", versioned=unresolved_var)  # Compliant


versioning = True
bucket = s3.Bucket(self, "MyUnversionedBucket", versioned=versioning)  # Compliant

value = "FALSE"
bucket = s3.Bucket(self, "MyUnversionedBucket", versioned=value)  # Compliant


a = function_call_for_coverage(10)


def test():
    bucket_versioning = False
#   ^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
    second_versioning = bucket_versioning
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
    bucket = s3.Bucket(self, "MyUnversionedBucket",
                       versioned=second_versioning  # Noncompliant {{Make sure an unversioned S3 bucket is safe here.}}
                    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       )
