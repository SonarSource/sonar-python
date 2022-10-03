from aws_cdk import aws_ec2 as ec2

### CfnSecurityGroupIngress
# A typical call to aws_cdk.aws_ec2.CfnSecurityGroupIngress would look like this. For simplicity/readability sake, we will omit arguments non-related to the check.
ec2.CfnSecurityGroupIngress(
    self,
    "ingress-all-ipv4-tcp-http",
    ip_protocol="6",
    cidr_ip="0.0.0.0/0",
    from_port=80,
    to_port=80,
    group_id=security_group.attr_group_id
)

# Sensitive test cases
ec2.CfnSecurityGroupIngress(
    ip_protocol="tcp",
    cidr_ip="0.0.0.0/0", # Noncompliant
    from_port=22,
    to_port=22
)
ec2.CfnSecurityGroupIngress(
    ip_protocol="6",
    cidr_ipv6="::/0", # Noncompliant
    from_port=22,
    to_port=22
)
ec2.CfnSecurityGroupIngress(
    ip_protocol="-1",
    cidr_ip="0.0.0.0/0" # Noncompliant
)
ec2.CfnSecurityGroupIngress(
    ip_protocol="-1",
    cidr_ipv6="::/0" # Noncompliant
)

# Sesntive "deported" test cases
badCidrIpV4 = "0.0.0.0/0"
ec2.CfnSecurityGroupIngress(
    ip_protocol="tcp",
    cidr_ip=badCidrIpV4, # Noncompliant
    from_port=22,
    to_port=22
)

# Compliant test cases
ec2.CfnSecurityGroupIngress( # not a bad protocol
    ip_protocol="7",
    cidr_ip="0.0.0.0/0",
    from_port=22,
    to_port=22
)
ec2.CfnSecurityGroupIngress( # has a valid IP address
    ip_protocol="6",
    cidr_ip="192.168.1.254/24",
    from_port=22,
    to_port=22
)
ec2.CfnSecurityGroupIngress( # has a valid IPv6 address
    ip_protocol="6",
    cidr_ipv6="1.2.3.4.5.6/24",
    from_port=22,
    to_port=22
)
ec2.CfnSecurityGroupIngress( # not within admin port
    ip_protocol="6",
    cidr_ip="0.0.0.0/0",
    from_port=25,
    to_port=30
)
ec2.CfnSecurityGroupIngress( # -1 protocol but has a valid ip address
    ip_protocol="-1",
    cidr_ip="192.168.1.254/24"
)
ec2.CfnSecurityGroupIngress( # -1 protocol but has a valid ipv6 address
    ip_protocol="-1",
    cidr_ipv6="1.2.3.4.5.6/24"
)
