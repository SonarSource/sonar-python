import aws_cdk.aws_dms as dms


class PublicNetworkAccessDatabaseMigration:
    def __init__(self, publicly_accessible_from_argument):
        dms.CfnReplicationInstance(
            self,
            "explicit_public",
            replication_instance_class="dms.t2.micro",
            allocated_storage=5,
            publicly_accessible=True,   # NonCompliant{{Make sure allowing public network access is safe here.}}
            replication_subnet_group_identifier=subnet_group.replication_subnet_group_identifier,
            vpc_security_group_ids=[vpc.vpc_default_security_group]
        )

        dms.CfnReplicationInstance(
            self,
            "explicit_private",
            replication_instance_class="dms.t2.micro",
            allocated_storage=5,
            publicly_accessible=False,
            replication_subnet_group_identifier=subnet_group.replication_subnet_group_identifier,
            vpc_security_group_ids=[vpc.vpc_default_security_group]
        )

        dms.CfnReplicationInstance(     # NonCompliant{{Make sure allowing public network access is safe here.}}
            self,
            "explicit_public"
        )

        dms.CfnReplicationInstance(
            self,
            "explicit_public",
            publicly_accessible=publicly_accessible_from_argument
        )

        access_true = True
#       ^^^^^^^^^^^^^^^^^^>
        dms.CfnReplicationInstance(
            self,
            "explicit_public",
            publicly_accessible=access_true  # NonCompliant{{Make sure allowing public network access is safe here.}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )

        access_false = False
        dms.CfnReplicationInstance(
            self,
            "explicit_public",
            publicly_accessible=access_false
        )

        dms.CfnReplicationInstance(
            self,
            "explicit_public",
            publicly_accessible=unknown
        )

        dms.CfnReplicationInstance(
            self,
            "explicit_public",
            publicly_accessible=getValue()
        )
