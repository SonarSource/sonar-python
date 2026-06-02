import aws_cdk.aws_apigateway as apigateway

class PublicApiIsSecuritySensitiveCfnMethodCheck:
    def __init__(self, auth_from_argument):
        apigateway.CfnMethod(
            self,
            "no-auth",
            authorization_type="NONE"  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        apigateway.CfnMethod(
            self,
            "no-auth-omit"
        )  # Compliant - omission no longer flagged

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
            authorization_type=auth_from_argument  # Compliant
        )

        apigateway.CfnMethod(
            self,
            "auth",
            authorization_type="COGNITO_USER_POOLS"  # Compliant
        )

        auth_type_none = "NONE"
#       ^^^^^^^^^^^^^^^^^^^^^^^>
        apigateway.CfnMethod(
            self,
            "no-auth",
            authorization_type=auth_type_none  # NonCompliant{{Ensure this API route requires authentication.}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )

        apigateway.CfnMethod(
            self,
            "auth",
            authorization_type=unknown
        )

        apigateway.CfnMethod(
            self,
            "auth",
            authorization_type=getValue()
        )
