import aws_cdk.aws_apigatewayv2 as apigateway

class PublicApiIsSecuritySensitiveCfnRouteCheck:
    def __init__(self, auth_from_argument):

        apigateway.CfnRoute(  # NonCompliant{{Omitting "authorization_type" disables authentication. Make sure it is safe here.}}
            self,
            "default-no-auth"
        )

        apigateway.CfnRoute(
            self,
            "no-auth",
            authorization_type="NONE"  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )

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
            authorization_type=auth_type_none  # NonCompliant{{Make sure that creating public APIs is safe here.}}
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
