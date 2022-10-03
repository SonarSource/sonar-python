from aws_cdk import aws_ec2 as ec2

### CfnSecurityGroup
# A typical call to aws_cdk.aws_ec2.CfnSecurityGroup would look like this. For simplicity/readability sake, we will omit arguments non-related to the check.
ec2.CfnSecurityGroup(self, "cfn-based-security-group", group_description="cfn based security group", group_name="cfn-based-security-group", vpc_id=vpc.vpc_id,
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="0.0.0.0/0", # Noncompliant
            from_port=22,
            to_port=22
        )
    ]
)

# Sensitive test cases
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="-1",
            cidr_ip="0.0.0.0/0" # Noncompliant
        )
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="-1",
            cidr_ipv6="::/0" # Noncompliant
        )
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="0.0.0.0/0", # Noncompliant
            from_port=10,
            to_port=30
        )
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="tcp",
            cidr_ip="0.0.0.0/0", # Noncompliant
            from_port=3300,
            to_port=3400
        )
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ipv6="::/0", # Noncompliant
            from_port=10,
            to_port=30
        )
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="tcp",
            cidr_ipv6="::/0", # Noncompliant
            from_port=3300,
            to_port=3400
        )
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="0.0.0.0/0", # Noncompliant
            from_port=22,
            to_port=22
        )
    ]
)


ec2.CfnSecurityGroup(
    security_group_ingress=[
        {
            "ipProtocol"    : "-1",
            "cidrIp"        : "0.0.0.0/0" # Noncompliant
        }
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        {
            "ipProtocol"    : "-1",
            "cidrIpv6"      : "::/0" # Noncompliant
        }
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        {
            "ipProtocol"    : "6",
            "cidrIp"        : "0.0.0.0/0", # Noncompliant
            "fromPort"      : 10,
            "toPort"        : 30
        }
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        {
            "ipProtocol"    : "tcp",
            "cidrIp"        : "0.0.0.0/0", # Noncompliant
            "fromPort"      : 3300,
            "toPort"        : 3400
        }
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        {
            "ipProtocol"    : "6",
            "cidrIpv6"      : "::/0", # Noncompliant
            "fromPort"      : 10,
            "toPort"        : 30
        }
    ]
)
ec2.CfnSecurityGroup(
    security_group_ingress=[
        {
            "ipProtocol"    : "tcp",
            "cidrIpv6"      : "::/0", # Noncompliant
            "fromPort"      : 3300,
            "toPort"        : 3400
        }
    ]
)

ec2.CfnSecurityGroup(
    self,
    "cfn-based-security-group",
    group_description="cfn based security group",
    group_name="cfn-based-security-group",
    vpc_id=vpc.vpc_id,
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="0.0.0.0/0", # Noncompliant
            from_port=22,
            to_port=22
        ),
        {
            "ipProtocol":"-1",
            "cidrIpv6":"::/0" # Noncompliant
        },
        { # Compliant
            "ipProtocol":"6",
            "cidrIp":"192.0.2.0/24",
            "fromPort":22,
            "toPort":22
        }
    ]
)

# Sensitive case with deported empty ip address
emptyIpV4 = "0.0.0.0/0"
ec2.CfnSecurityGroup(
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="-1",
            cidr_ip=emptyIpV4 # Noncompliant
        )
    ]
)

# Sensitive case with deported invalid arguments in array
arrayInvalid = [
   ec2.CfnSecurityGroup.IngressProperty(
       ip_protocol="-1",
       cidr_ip="0.0.0.0/0" # Noncompliant
   )
]
ec2.CfnSecurityGroup(
    security_group_ingress=arrayInvalid
)

# Sensitive case with deported IngressProperty object
badIngressProperty = ec2.CfnSecurityGroup.IngressProperty(
    ip_protocol="-1",
    cidr_ip="0.0.0.0/0" # Noncompliant
)
ec2.CfnSecurityGroup(
    security_group_ingress=[badIngressProperty]
)

# Not reporting below case as sensitive : the object is not used anywhere
badIngressPropertyUnused = ec2.CfnSecurityGroup.IngressProperty(
    ip_protocol="-1",
    cidr_ip="0.0.0.0/0"
)

# Compliant test cases
ec2.CfnSecurityGroup( # has a valid IP address
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="192.0.2.0/0",
            from_port=22,
            to_port=22
        )
    ]
)
ec2.CfnSecurityGroup( # from/to port does not contain admin port
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="0.0.0.0/0",
            from_port=25,
            to_port=26
        )
    ]
)
ec2.CfnSecurityGroup( # does not use a bad protocol
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="7",
            cidr_ip="0.0.0.0/0",
            from_port=22,
            to_port=22
        )
    ]
)
ec2.CfnSecurityGroup( # has a valid IPv6 address
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="7",
            cidr_ipv6="1.2.3.4.5.6/24",
            from_port=22,
            to_port=22
        )
    ]
)
ec2.CfnSecurityGroup( # has a valid IP address with -1 protocol
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="-1",
            cidr_ip="192.168.1.254/24"
        )
    ]
)
ec2.CfnSecurityGroup( # has a valid IPv6 address with -1 protocol
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="-1",
            cidr_ipv6="1.2.3.4.5.6/24"
        )
    ]
)

ec2.CfnSecurityGroup( # has a valid IP address
    security_group_ingress=[
        {
            "ipProtocol":"6",
            "cidrIp":"192.0.2.0/24",
            "fromPort":22,
            "toPort":22
        }
    ]
)
ec2.CfnSecurityGroup( # from/to port does not contain admin port
    security_group_ingress=[
        {
            "ipProtocol":"6",
            "cidrIp":"0.0.0.0/0",
            "fromPort":25,
            "toPort":30
        }
    ]
)
ec2.CfnSecurityGroup( # does not use a bad protocol
    security_group_ingress=[
        {
            "ipProtocol":"7",
            "cidrIp":"0.0.0.0/0",
            "fromPort":22,
            "toPort":22
        }
    ]
)
ec2.CfnSecurityGroup( # has a valid IPv6 address
    security_group_ingress=[
        {
            "ipProtocol":"6",
            "cidrIpv6":"1.2.3.4.5.6/24",
            "fromPort":22,
            "toPort":22
        }
    ]
)
ec2.CfnSecurityGroup( # has a valid IP address with -1 protocol
    security_group_ingress=[
        {
            "ipProtocol":"-1",
            "cidrIp":"192.168.1.254/24"
        }
    ]
)
ec2.CfnSecurityGroup( # has a valid IPv6 address with -1 protocol
    security_group_ingress=[
        {
            "ipProtocol":"-1",
            "cidrIpv6":"1.2.3.4.5.6/24"
        }
    ]
)
ec2.CfnSecurityGroup( # incomplete : missong to_port
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="0.0.0.0/0",
            from_port=10
        )
    ]
)
ec2.CfnSecurityGroup( # incomplete : missong from_port
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty(
            ip_protocol="6",
            cidr_ip="0.0.0.0/0",
            to_port=30
        )
    ]
)
ec2.CfnSecurityGroup( # empty array
    security_group_ingress=[]
)
ec2.CfnSecurityGroup( # dictionary instead
    security_group_ingress={}
)
ec2.CfnSecurityGroup( # no argument passed to the call expression
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty()
    ]
)
ec2.CfnSecurityGroup( # not a call expression
    security_group_ingress=[
        ec2.CfnSecurityGroup.IngressProperty
    ]
)
