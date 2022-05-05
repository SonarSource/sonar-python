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

        public_access_only_block_acls_by_reference = s3.BlockPublicAccess.BLOCK_ACLS
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
        bucket = s3.Bucket(self, "PublicAccessOnlyBlockAclsByReference",
                           block_public_access=public_access_only_block_acls_by_reference)  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
    #                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

        referenced_unblock_public_accesses = s3.BlockPublicAccess(
            block_public_acls=True,
            ignore_public_acls=False,  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
            block_public_policy=True,
            restrict_public_buckets=True)

        bucket = s3.Bucket(self, "ReferencedUnblockPublicAccesses",
                           block_public_access=referenced_unblock_public_accesses)

        referenced_value_unblock_public_accesses = False
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Propagated setting.}}
        bucket = s3.Bucket(self, "ReferencedValueUnblockPublicAccesses",
                           block_public_access=s3.BlockPublicAccess(
                               block_public_acls=referenced_value_unblock_public_accesses,  # NonCompliant {{Make sure allowing public ACL/policies to be set is safe here.}}
                           #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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

        referenced_block_public_accesses = s3.BlockPublicAccess(
            block_public_acls=True,
            ignore_public_acls=True,
            block_public_policy=True,
            restrict_public_buckets=True)

        bucket = s3.Bucket(self, "ReferencedBlockPublicAccesses",
                           block_public_access=referenced_block_public_accesses)

        unused_block_public_access = s3.BlockPublicAccess(
            block_public_acls=True,
            ignore_public_acls=False,
            block_public_policy=True,
            restrict_public_buckets=True)

        bucket = s3.Bucket(self, "UnresolvedReferencedBlockPublicAccesses",
                           block_public_access=unknown_block_public_access)

        loop_a = loop_b
        loop_b = loop_a

        bucket = s3.Bucket(self, "LoopReferenceBlockPublicAccess",
                           block_public_access=loop_a)
