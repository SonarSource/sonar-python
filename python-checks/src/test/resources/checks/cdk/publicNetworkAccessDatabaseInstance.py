from aws_cdk import aws_rds as rds
from aws_cdk import aws_ec2 as ec2

rds.DatabaseInstance(
    publicly_accessible=True,   # Noncompliant {{Make sure allowing public network access is safe here.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^
    vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)
)

rds.DatabaseInstance(
    publicly_accessible=False,  # Compliant
    vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)
)

rds.DatabaseInstance(
    publicly_accessible=True,  # Compliant, IP won't be routable
    vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT),
)

rds.DatabaseInstance(
    publicly_accessible=True,  # Noncompliant
)

rds.DatabaseInstance(
    # Noncompliant@+1 {{Make sure allowing public network access is safe here.}}
    vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
#                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
)

rds.DatabaseInstance(
    publicly_accessible=True,  # Noncompliant
    vpc_subnets=unknown_subnets
)

vpc = ec2.Vpc()

rds_subnet_group_public = rds.CfnDBSubnetGroup(
    subnet_ids=vpc.select_subnets(
        subnet_type=ec2.SubnetType.PUBLIC
    ).subnet_ids
)

rds.CfnDBInstance(
    publicly_accessible=True, # Noncompliant
    db_subnet_group_name=rds_subnet_group_public.ref
)

rds.CfnDBInstance(
    publicly_accessible=False, # Ok
    db_subnet_group_name=rds_subnet_group_public.ref
)

rds_subnet_group_private = rds.CfnDBSubnetGroup(
    subnet_ids=vpc.select_subnets(
        subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT
    ).subnet_ids
)

rds.CfnDBInstance(
    publicly_accessible=True, # Ok
    db_subnet_group_name=rds_subnet_group_private.ref
)

rds.CfnDBInstance(
    publicly_accessible=True, # Noncompliant
    db_subnet_group_name=unknown_subnet_group_name
)

rds.CfnDBInstance(
    db_subnet_group_name=unknown_subnet_group_name # Ok, publicly_accessible is not `True`
)

rds.CfnDBInstance(
    publicly_accessible=True, # Noncompliant
    db_subnet_group_name=unknown_subnet_group_name.whatever
)

rds_subnet_group_unknown = rds.CfnDBSubnetGroup()

rds.CfnDBInstance(
    publicly_accessible=True, # Noncompliant
    db_subnet_group_name=rds_subnet_group_unknown.ref
)

rds_subnet_group_unknown_with_subnet_ids = rds.CfnDBSubnetGroup(
    subnet_ids=unknown_vpc.whatever
)

rds.CfnDBInstance(
    publicly_accessible=True, # Noncompliant
    db_subnet_group_name=rds_subnet_group_unknown_with_subnet_ids.ref
)

rds_subnet_group_unknown_with_subnet_ids_from_variable = rds.CfnDBSubnetGroup(
    subnet_ids=get_subnets_from_somewhere.subnet_ids
)

rds.CfnDBInstance(
    publicly_accessible=True, # Noncompliant
    db_subnet_group_name=rds_subnet_group_unknown_with_subnet_ids_from_variable.ref
)
