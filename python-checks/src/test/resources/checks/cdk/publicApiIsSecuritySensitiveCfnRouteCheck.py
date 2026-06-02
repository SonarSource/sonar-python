import aws_cdk.aws_apigatewayv2 as apigateway

class PublicApiIsSecuritySensitiveCfnRouteCheck:
    def __init__(self, auth_from_argument):

        apigateway.CfnRoute(
            self,
            "no-auth",
            authorization_type="NONE"  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        apigateway.CfnRoute(
            self,
            "default-no-auth"
        )  # Compliant - omission no longer flagged

        apigateway.CfnRoute(
            self,
            "auth",
            authorization_type="AWS_IAM"  # Compliant
        )

        apigateway.CfnRoute(
            self,
            "auth",
            authorization_type=auth_from_argument  # Compliant
        )

        auth_type_none = "NONE"
#       ^^^^^^^^^^^^^^^^^^^^^^^>
        apigateway.CfnRoute(
            self,
            "no-auth",
            authorization_type=auth_type_none  # NonCompliant{{Ensure this API route requires authentication.}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )

        apigateway.CfnRoute(
            self,
            "auth",
            authorization_type=unknown
        )

        apigateway.CfnRoute(
            self,
            "auth",
            authorization_type=getValue()
        )
