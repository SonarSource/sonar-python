from aws_cdk import aws_s3 as s3
from aws_cdk import Stack
from constructs import Construct


class NonCompliantStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # NonCompliant@+1 {{No Public Access Block configuration prevents public ACL/policies to be set on this S3 bucket. Make sure it is safe here.}}
        bucket = s3.Bucket(self, "PublicAccessIsNotBlockedByDefault")
        #        ^^^^^^^^^

        # NonCompliant@+2 {{Make sure allowing public ACL/policies to be set is safe here.}}
        bucket = s3.Bucket(self, "PublicAccessOnlyBlockAcls",
                           block_public_access=s3.BlockPublicAccess.BLOCK_ACLS)

        block_public_access = s3.BlockPublicAccess.BLOCK_ACLS
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
        bucket = s3.Bucket(self, "PublicAccessOnlyBlockAclsByReference",
                           block_public_access=block_public_access)  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
    #                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        bucket = s3.Bucket(self, "AllowPublicReadAccess",
                           public_read_access=True)  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}

        public_read_access = True
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
        bucket = s3.Bucket(self, "AllowPublicReadAccessByReference",
                           public_read_access=public_read_access)  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
    #                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        bucket = s3.Bucket(self, "SingleUnblockPublicAccesses",
                           block_public_access=s3.BlockPublicAccess(
                               block_public_acls=False,  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
                           #   ^^^^^^^^^^^^^^^^^^^^^^^
                               ignore_public_acls=True,
                               block_public_policy=True,
                               restrict_public_buckets=True))

        bucket = s3.Bucket(self, "MultipleUnblockPublicAccesses",
                           block_public_access=s3.BlockPublicAccess(
                               block_public_acls=False,  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
                           #   ^^^^^^^^^^^^^^^^^^^^^^^
                               ignore_public_acls=True,
                               block_public_policy=False,  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
                           #   ^^^^^^^^^^^^^^^^^^^^^^^^^
                               restrict_public_buckets=True))

        blockPublicAccess = s3.BlockPublicAccess(
            block_public_acls=True,
            ignore_public_acls=False,  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
            block_public_policy=True,
            restrict_public_buckets=True)

        bucket = s3.Bucket(self, "ReferencedUnblockPublicAccesses",
                           block_public_access=blockPublicAccess)

        block_public_acls = False
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
        bucket = s3.Bucket(self, "ReferencedValueUnblockPublicAccesses",
                           block_public_access=s3.BlockPublicAccess(
                               block_public_acls=block_public_acls,  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
                           #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                               ignore_public_acls=True,
                               block_public_policy=True,
                               restrict_public_buckets=True))


class CompliantStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(self, "PublicAccessOnlyBlockAll",
                           block_public_access=s3.BlockPublicAccess.BLOCK_ALL)

        bucket = s3.Bucket(self, "AllBlockPublicAccesses",
                           block_public_access=s3.BlockPublicAccess(
                               block_public_acls=True,
                               ignore_public_acls=True,
                               block_public_policy=True,
                               restrict_public_buckets=True))

        blockPublicAccess = s3.BlockPublicAccess(
            block_public_acls=True,
            ignore_public_acls=True,
            block_public_policy=True,
            restrict_public_buckets=True)

        bucket = s3.Bucket(self, "ReferencedBlockPublicAccesses",
                           block_public_access=blockPublicAccess)

        unusedBlockPublicAccess = s3.BlockPublicAccess(
            block_public_acls=True,
            ignore_public_acls=False,
            block_public_policy=True,
            restrict_public_buckets=True)

        bucket = s3.Bucket(self, "UnresolvedReferencedBlockPublicAccesses",
                           block_public_access=unknownblockPublicAccess)
