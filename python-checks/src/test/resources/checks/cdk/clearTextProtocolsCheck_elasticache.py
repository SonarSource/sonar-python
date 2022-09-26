import aws_cdk.aws_elasticache as elasticache


class NonCompliantStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:

        # Noncompliant@+1 {{Make sure that disabling transit encryption is safe here.}}
        elasticache.CfnReplicationGroup(transit_encryption_enabled=False)
        # Noncompliant@+1 {{Omitting `transit_encryption_enabled` causes transit encryption to be disabled. Make sure it is safe here.}}
        elasticache.CfnReplicationGroup()

        elasticache.CfnReplicationGroup(transit_encryption_enabled=True)
        elasticache.CfnReplicationGroup(transit_encryption_enabled=transit_encryption_enabled)
