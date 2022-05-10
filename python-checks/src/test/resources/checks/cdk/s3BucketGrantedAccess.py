from aws_cdk import aws_s3 as s3, aws_s3_deployment as s3deploy


bucket = s3.Bucket(self, "bucket")  # Compliant by default
bucket.grant_public_access()  # NonCompliant

def grant_noncompliant():
    s1 = s3.Bucket(self, "BucketToDeploy")
    s2 = s1
    s2.grant_public_access()  # NonCompliant {{Make sure allowing unrestricted access to objects from this bucket is safe here.}}
#   ^^^^^^^^^^^^^^^^^^^^^^

    foo = Foo()
    # FP : due to the fact we check whether aws_cdk is imported and the below function is called
    foo.grant_public_access() # NonCompliant {{Make sure allowing unrestricted access to objects from this bucket is safe here.}}

bucket = s3.Bucket(self, "bucket",
                   access_control=s3.bar.WHAT_EVER       # Compliant
                   )

bucket = s3.Bucket(self, "bucket",
                   access_control=s3.BucketAccessControl.PRIVATE       # Compliant
                   )

bucket = s3.Bucket(self, "bucket",
                   access_control=s3.BucketAccessControl.PUBLIC_READ_WRITE     # NonCompliant {{Make sure granting PUBLIC_READ_WRITE access is safe here.}}
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   )

bucket = s3.Bucket(self, "bucket",
                   access_control=s3.BucketAccessControl.PUBLIC_READ     # NonCompliant {{Make sure granting PUBLIC_READ access is safe here.}}
                   )

bucket = s3.Bucket(self, "bucket",
                   access_control=s3.BucketAccessControl.AUTHENTICATED_READ     # NonCompliant {{Make sure granting AUTHENTICATED_READ access is safe here.}}
                   )

def create_public_bucket():
    control1 = s3.BucketAccessControl.PUBLIC_READ
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
    bucket = s3.Bucket(self, "bucket",
                       access_control=control1     # NonCompliant {{Make sure granting PUBLIC_READ access is safe here.}}
#                      ^^^^^^^^^^^^^^^^^^^^^^^
                       )


bucket_to_deplay = s3.Bucket(self, "BucketToDeploy")

s3deploy.BucketDeployment(self, "Deploy",                                   # Compliant
                          sources=[s3deploy.Source.asset("./deploy-dist")],
                          destination_bucket=bucket_to_deploy
)

s3deploy.BucketDeployment(self, "Deploy2",
                          destination_bucket=bucket_to_deploy,
                          access_control=s3.BucketAccessControl.PUBLIC_READ_WRITE     # NonCompliant {{Make sure granting PUBLIC_READ_WRITE access is safe here.}}
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          )

s3deploy.BucketDeployment(self, "Deploy3",
                          destination_bucket=bucket_to_deploy,
                          access_control=s3.BucketAccessControl.PRIVATE       # Compliant
                          )

