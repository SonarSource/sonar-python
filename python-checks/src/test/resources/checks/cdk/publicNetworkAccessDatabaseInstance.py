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

