from aws_cdk import aws_rds as rds
from aws_cdk import aws_ec2 as ec2

rds.DatabaseInstance(
    publicly_accessible=True,   # Noncompliant
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
