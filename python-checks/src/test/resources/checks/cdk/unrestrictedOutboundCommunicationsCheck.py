import aws_cdk.aws_ec2 as ec2

class UnrestrictedOutbound:
    def __init__(self, vpc):
        ec2.SecurityGroup(
            self,
            "SensitiveExplicit",
            "sg-1234",
            allow_all_outbound=True  # NonCompliant{{Make sure that allowing unrestricted outbound communications is safe here.}}
        )

        ec2.SecurityGroup(  # NonCompliant{{Omitting "allow_all_outbound" enables unrestricted outbound communications. Make sure it is safe here.}}
            self,
            "SensitiveDefault",
            vpc=vpc
        )

        ec2.SecurityGroup.from_security_group_id(
            self,
            "SensitiveExplicit",
            "sg-1234",
            allow_all_outbound=True  # NonCompliant{{Make sure that allowing unrestricted outbound communications is safe here.}}
        )
 
        ec2.SecurityGroup.from_security_group_id(
            self,
            "SensitiveExplicit",
            "sg-1234",
            allow_all_outbound=False
        )

        ec2.SecurityGroup.from_security_group_id(  # Compliant ref: SONARPY-1159 and SONARPY-1419
            self,
            "SensitiveDefault",
            vpc=vpc
        )

        outbound_restriction=True
#       ^^^^^^^^^^^^^^^^^^^^^^^^^>

        ec2.SecurityGroup(
            self,
            "SensitiveExplicit",
            "sg-1234",
            allow_all_outbound=outbound_restriction  # NonCompliant{{Make sure that allowing unrestricted outbound communications is safe here.}}
        #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )
        ec2.SecurityGroup(
            self,
            "SensitiveExplicit",
            "sg-1234",
            allow_all_outbound=unknown
        )

        ec2.SecurityGroup(
            self,
            "SensitiveExplicit",
            "sg-1234",
            allow_all_outbound=getValue()
        )
