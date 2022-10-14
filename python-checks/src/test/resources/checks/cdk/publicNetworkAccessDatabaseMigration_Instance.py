import aws_cdk.aws_ec2 as ec2

class Test(Stack):
    def test(self):
        ## aws_cdk.aws_ec2.Instance
        # A normal call to aws_cdk.aws_ec2.Instance look like this, for the sake of simplicity/readability, non-checked parameters will be omitted in test
        obj = ec2.Instance(
            self,
            "vpc_subnet_public",
            instance_type=nano_t2,
            machine_image=ec2.MachineImage.latest_amazon_linux(),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT)
        )

        # Sensitive
        ec2.Instance(vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)) # Noncompliant{{Make sure allowing public network access is safe here.}}
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ec2.Instance(vpc_subnets={"subnet_type" : ec2.SubnetType.PUBLIC}) # Noncompliant
#                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Compliant
        ec2.Instance(vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT))
        ec2.Instance(vpc_subnets={"subnet_type" : ec2.SubnetType.PRIVATE_WITH_NAT})
        ec2.Instance(vpc_subnets=ec2.SubnetSelection(subnet_type="ec2.SubnetType.PUBLIC"))
        ec2.Instance(vpc_subnets={"subnet_type" : "ec2.SubnetType.PUBLIC"})
        ec2.Instance(vpc_subnets=ec2.SubnetSelection(random_attribute=ec2.SubnetType.PUBLIC))
        ec2.Instance(vpc_subnets={"random_attribute" : ec2.SubnetType.PUBLIC})

        ## aws_cdk.aws_ec2.CfnInstance
        # A normal call to aws_cdk.aws_ec2.CfnInstance look like this, for the sake of simplicity/readability, non-checked parameters will be omitted in test
        ec2.CfnInstance(
            self,
            "cfn_private",
            instance_type="t2.micro",
            image_id="ami-0ea0f26a6d50850c5",
            network_interfaces=[
                ec2.CfnInstance.NetworkInterfaceProperty(
                    device_index="0",
                    associate_public_ip_address=False, # Compliant
                    delete_on_termination=True,
                    subnet_id=vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT).subnet_ids[0]
                )
            ]
        )

        # Sensitive
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True)]) # Noncompliant{{Make sure allowing public network access is safe here.}}
#                                                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True}]) # Noncompliant
#                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PUBLIC).subnet_ids[0])]) # Noncompliant
#                                                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True, "subnet_id" : ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PUBLIC).subnet_ids[0]}]) # Noncompliant
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=ec2.Vpc.select_subnets(subnet_type="random value").subnet_ids[0])]) # Noncompliant
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True, "subnet_id" : ec2.Vpc.select_subnets(subnet_type="random value")}]) # Noncompliant
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=ec2.Vpc.select_subnets(random_attribute=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0])]) # Noncompliant
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True, "subnet_id" : ec2.Vpc.select_subnets(random_attribute=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0]}]) # Noncompliant
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id="unknown"[0])]) # Noncompliant

        # Compliant
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=False)])
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : False}])
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address="any value")])
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : "any value"}])
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty()])
        ec2.CfnInstance(network_interfaces=[{}])
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(random_attribute=True)])
        ec2.CfnInstance(network_interfaces=[{"random_attribute" : True}])
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0])])
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS).subnet_ids[0])])
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT).subnet_ids[0])])
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True, "subnet_id" : ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0]}])
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True, "subnet_id" : ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS).subnet_ids[0]}])
        ec2.CfnInstance(network_interfaces=[{"associate_public_ip_address" : True, "subnet_id" : ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT).subnet_ids[0]}])

        subnet_id = ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids[0]
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=subnet_id)])
        subnet_ids = ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=subnet_ids[0])])
        subnet = ec2.Vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)
        ec2.CfnInstance(network_interfaces=[ec2.CfnInstance.NetworkInterfaceProperty(associate_public_ip_address=True, subnet_id=subnet.subnet_ids[0])])

