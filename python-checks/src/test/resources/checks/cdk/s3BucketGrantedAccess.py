from aws_cdk import aws_s3 as s3, aws_s3_deployment as s3deploy


bucket = s3.Bucket(self, "bucket")  # Compliant by default
# bucket.grant_public_access() # NonCompliant


bucket = s3.Bucket(self, "bucket",
                   access_control=s3.BucketAccessControl.PRIVATE       # Compliant
                   )

bucket = s3.Bucket(self, "bucket",
                   access_control=s3.BucketAccessControl.PUBLIC_READ_WRITE     # NonCompliant {{Make sure granting access to [AllUsers|AuthenticatedUsers] group is safe here.}}
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   )

bucket = s3.Bucket(self, "bucket",
                   access_control=s3.BucketAccessControl.PUBLIC_READ     # NonCompliant
                   )

def create_public_bucket():
    control1 = s3.BucketAccessControl.PUBLIC_READ
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
    bucket = s3.Bucket(self, "bucket",
                       access_control=control1     # NonCompliant {{Make sure granting access to [AllUsers|AuthenticatedUsers] group is safe here.}}
#                      ^^^^^^^^^^^^^^^^^^^^^^^
                       )


bucket_to_deplay = s3.Bucket(self, "BucketToDeploy")

s3deploy.BucketDeployment(self, "Deploy",                                   # Compliant
                          sources=[s3deploy.Source.asset("./deploy-dist")],
                          destination_bucket=bucket_to_deploy
)

s3deploy.BucketDeployment(self, "Deploy2",
                          destination_bucket=bucket_to_deploy,
                          access_control=s3.BucketAccessControl.PUBLIC_READ_WRITE     # NonCompliant
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          )

s3deploy.BucketDeployment(self, "Deploy3",
                          destination_bucket=bucket_to_deploy,
                          access_control=s3.BucketAccessControl.PRIVATE       # Compliant
                          )
