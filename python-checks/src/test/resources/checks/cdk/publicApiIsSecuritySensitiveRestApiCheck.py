import aws_cdk.aws_apigateway as apigateway


class PublicApiIsSecuritySensitiveRestApiCheck:
    def __init__(self, auth_from_argument):
        api = apigateway.RestApi(
            self,
            "RestApi"
        )

        test = api.root.add_resource("test")
        test.add_method(
            "GET",
            authorization_type=apigateway.AuthorizationType.NONE  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )

        test.add_method(
            "POST",
            authorization_type=apigateway.AuthorizationType.IAM  # Compliant
        )

        auth_type_none = apigateway.AuthorizationType.NONE
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
        test.add_method(
            "GET",
            authorization_type=auth_type_none  # NonCompliant{{Make sure that creating public APIs is safe here.}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )

        auth_type_cognito = apigateway.AuthorizationType.COGNITO
        test.add_method(
            "GET",
            authorization_type=auth_type_cognito
        )

        test_child = test.add_resource("testchild")

        test_child.parent_resource.add_method(
            "HEAD",
            authorization_type=apigateway.AuthorizationType.NONE  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )

        test.get_resource("testchild").add_method(
            "PATCH",
            authorization_type=apigateway.AuthorizationType.NONE  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )

        test.api.root.add_method(  # FN
            "DELETE"
        )


class PublicApiIsSecuritySensitiveRestApiSecureConstructorCheck:
    def __init__(self, auth_from_argument):
        api = apigateway.RestApi(
            self,
            "RestApiDefault",
            default_method_options={"authorization_type": apigateway.AuthorizationType.IAM}
        )

        test = api.root.add_resource("test")
        test.api.root.add_method(  # Compliant; secure default
            "DELETE"
        )

class PublicApiIsSecuritySensitiveRestApiSecureAddResourceCallCheck:
    def __init__(self, auth_from_argument):
        api = apigateway.RestApi(
            self,
            "RestApiDefault"
        )

        opts = apigateway.MethodOptions(
            authorization_type=apigateway.AuthorizationType.IAM
        )

        test = api.root.add_resource(
            "test",
            default_method_options=opts
        )
        test.add_method(  # Compliant because of default_method_options
            "GET"
        )
