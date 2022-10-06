import aws_cdk.aws_apigateway as apigateway

class PublicApiIsSecuritySensitiveCfnMethodCheck:
    def __init__(self, vpc):
        apigateway.CfnMethod(
            self,
            "no-auth",
            authorization_type="NONE"  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )

        apigateway.CfnMethod(
            self,
            "auth",
            authorization_type="AWS_IAM"  # Compliant
        )

        apigateway.CfnMethod(
            self,
            "auth",
            authorization_type="CUSTOM"  # Compliant
        )

        apigateway.CfnMethod(
            self,
            "auth",
            authorization_type="COGNITO_USER_POOLS"  # Compliant
        )

        apigateway.CfnMethod(   # NonCompliant{{Omitting "authorization_type" disables authentication. Make sure it is safe here.}}
            self,
            "no-auth"
        )

        auth_type_none = "NONE"
#       ^^^^^^^^^^^^^^^^^^^^^^^>
        apigateway.CfnMethod(
            self,
            "no-auth",
            authorization_type=auth_type_none  # NonCompliant{{Make sure that creating public APIs is safe here.}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        )
