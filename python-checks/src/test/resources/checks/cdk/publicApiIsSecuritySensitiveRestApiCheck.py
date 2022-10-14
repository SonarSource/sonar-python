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

        test.api.root.add_method(  # NonCompliant{{Omitting "authorization_type" disables authentication. Make sure it is safe here.}}
            "DELETE"
        )

        test.api.root.add_method(
            "DELETE",
            authorization_type=auth_from_argument   # Compliant, value comes from the outside
        )

        test.api.root.add_method(
            "DELETE",
            authorization_type=unknown   # Compliant, value is unknown
        )


class PublicApiIsSecuritySensitiveRestApiSecureConstructorCheck:
    def __init__(self, auth_from_argument):
        api = apigateway.RestApi(
            self,
            "RestApiDefault",
            default_method_options={"authorization_type": apigateway.AuthorizationType.IAM}
        )

        test = api.root.add_resource("test")
        test.add_method("GET")  # Compliant; secure default
        test.add_method(
            "GET",
            authorization_type=apigateway.AuthorizationType.NONE  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )
        test.api.root.add_method(
            "DELETE",
            authorization_type=apigateway.AuthorizationType.NONE  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )
        test.api.root.add_method(  # Compliant; secure default
            "DELETE"
        )
        test_child = test.add_resource("test_child")
        test_child.parent_resource.add_method("HEAD")   # Compliant; secure default
        test_child.parent_resource.add_method(
            "HEAD",
            authorization_type=apigateway.AuthorizationType.NONE  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )
        test_child.parent_resource.add_method(
            "HEAD",
            authorization_type=auth_from_argument   # Compliant, value comes from the outside
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
        test.add_method(
            "GET",
            authorization_type=auth_from_argument   # Compliant, value comes from the outside
        )
        test.add_method(
            "GET",
            authorization_type=unknown   # Compliant, value is unknown
        )
        test.add_method(
            "GET",
            authorization_type=apigateway.AuthorizationType.NONE  # NonCompliant{{Make sure that creating public APIs is safe here.}}
        )
        test.add_method(
            "GET",
            authorization_type=apigateway.AuthorizationType.IAM  # Compliant
        )

class PublicApiIsSecuritySensitiveRestApiUnsecureConstructorDictionary:
    def __init__(self):
        api = apigateway.RestApi(
            self,
            "RestApiDefault",
            default_method_options={"authorization_type": apigateway.AuthorizationType.NONE}
        )

        test = api.root.add_resource("test")
        test.add_method("GET")  # NonCompliant{{Omitting "authorization_type" disables authentication. Make sure it is safe here.}}

class PublicApiIsSecuritySensitiveRestApiUnsecureConstructorMethodOptions:
    def __init__(self):
        opts = apigateway.MethodOptions(
            authorization_type=apigateway.AuthorizationType.NONE
        )
        api = apigateway.RestApi(
            self,
            "RestApiDefault",
            default_method_options=opts
        )

        test = api.root.add_resource("test")
        test.add_method("GET")  # NonCompliant{{Omitting "authorization_type" disables authentication. Make sure it is safe here.}}
