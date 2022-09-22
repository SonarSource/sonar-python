import aws_cdk.aws_efs as efs


class NonCompliantStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        efs.FileSystem(encrypted=False)  # Noncompliant {{Make sure that using unencrypted file systems is safe here.}}
    #                  ^^^^^^^^^^^^^^^

        efs.CfnFileSystem()  # Noncompliant {{Omitting "encrypted" disables EFS encryption. Make sure it is safe here.}}
    #   ^^^^^^^^^^^^^^^^^
        efs.CfnFileSystem(encrypted=False)  # Noncompliant
        efs.CfnFileSystem(encrypted=None)  # Noncompliant


class CompliantStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        efs.FileSystem(encrypted=True)
        efs.FileSystem(encrypted=encrypted)
        efs.FileSystem()

        efs.CfnFileSystem(encrypted=True)
        efs.CfnFileSystem(encrypted=encrypted)
