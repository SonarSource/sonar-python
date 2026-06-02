import aws_cdk.aws_apigateway as apigw


class GateOne:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # Gate 1: state-changing methods → NonCompliant
        api.root.add_resource("users").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
        api.root.add_resource("items").add_method(
            "PUT",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
        api.root.add_resource("items").add_method(
            "DELETE",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
        api.root.add_resource("items").add_method(
            "PATCH",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
        api.root.add_resource("items").add_method(
            "ANY",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Gate 1: safe (read-only) methods → Compliant
        api.root.add_resource("users").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - GET is not state-changing
        )
        api.root.add_resource("users").add_method(
            "HEAD",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - HEAD is not state-changing
        )
        api.root.add_resource("users").add_method(
            "OPTIONS",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - OPTIONS is not state-changing
        )

        # Non-NONE authorization → Compliant
        api.root.add_resource("users").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.IAM  # Compliant
        )
        api.root.add_resource("users").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.COGNITO  # Compliant
        )

        # NONE via variable → NonCompliant (secondary location on assignment)
        auth_type_none = apigw.AuthorizationType.NONE
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
        api.root.add_resource("users").add_method(
            "POST",
            authorization_type=auth_type_none  # NonCompliant{{Ensure this API route requires authentication.}}
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )

        # authorization_type from unknown external → Compliant
        api.root.add_resource("users").add_method(
            "POST",
            authorization_type=auth_from_argument  # Compliant - value comes from outside
        )

        # HTTP method from variable → Compliant (cannot prove method statically)
        api.root.add_resource("users").add_method(
            get_method(),
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - method not statically provable
        )

        # Omitting authorization_type entirely → Compliant (omission is not flagged)
        api.root.add_resource("users").add_method("POST")  # Compliant - omission not flagged


class Stage2Exclusion:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # Stage 2: known-public paths → Compliant even with state-changing methods
        api.root.add_resource("login").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("signup").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("register").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("healthcheck").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("status").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("token").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("callback").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("jwks").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("well-known").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource(".well-known").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - leading dot normalized
        )
        api.root.add_resource("authenticate").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("forgot-password").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("health-check").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )
        api.root.add_resource("public-keys").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - public endpoint
        )


class GateTwo:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # Gate 2: exact sensitive segment + GET → NonCompliant
        api.root.add_resource("admin").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
        api.root.add_resource("management").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
        api.root.add_resource("internal").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Gate 2: compound names are NOT exact matches → Compliant
        api.root.add_resource("admin-portal").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - not an exact segment match
        )
        api.root.add_resource("internal-api").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - not an exact segment match
        )
        api.root.add_resource("management-ui").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - not an exact segment match
        )

        # Gate 2 via variable assignment (ExpressionFlow resolves the receiver)
        admin_res = api.root.add_resource("admin")
        admin_res.add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Stage 2 exclusion takes priority over Gate 2 via variable assignment
        login_res = api.root.add_resource("login")
        login_res.add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - Stage 2 exclusion
        )

        # Neutral name with GET → no gate fires → Compliant
        api.root.add_resource("reports").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - neutral name, safe method
        )

        # Case-insensitive path matching
        api.root.add_resource("Login").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - case-insensitive Stage 2 exclusion
        )
        api.root.add_resource("Admin").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Qualifier is a call but not add_resource → path unresolvable → no Gate 2 → Compliant
        api.root.get_resource("admin").add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - receiver not add_resource, path unknown
        )

        # Variable assigned to non-add_resource call → path unresolvable → no Gate 2 → Compliant
        some_res = api.root.get_resource("admin")
        some_res.add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - receiver not add_resource, path unknown
        )


class PrivateEndpoint:
    def __init__(self):
        # Compliant: PRIVATE endpoint — VPC-only API, all methods suppressed
        private_api = apigw.RestApi(self, "PrivateApi",
            endpoint_configuration=apigw.EndpointConfiguration(
                types=[apigw.EndpointType.PRIVATE]
            )
        )
        private_api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - VPC-only API
        )
        private_api.root.add_resource("data").add_method(
            "DELETE",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - VPC-only API
        )

        # Compliant: PRIVATE with nested add_resource
        private_api.root.add_resource("a").add_resource("b").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - VPC-only API
        )

        # Compliant: PRIVATE via variable resource
        nested = private_api.root.add_resource("nested")
        nested.add_method(
            "DELETE",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - VPC-only API
        )

        # Public (EDGE) endpoint — no suppression, still raises
        public_api = apigw.RestApi(self, "PublicApi")
        public_api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # REGIONAL endpoint is not PRIVATE — still raises
        regional_api = apigw.RestApi(self, "RegionalApi",
            endpoint_configuration=apigw.EndpointConfiguration(
                types=[apigw.EndpointType.REGIONAL]
            )
        )
        regional_api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


class NestedResourceViaVariable:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # Variable holding intermediate add_resource, then chained add_resource on it
        # Exercises the resolveRestApi NAME→assigned-CALL_EXPR path.
        base = api.root.add_resource("base")
        base.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Same shape, Stage 2 exclusion applies through the resolved chain.
        base2 = api.root.add_resource("base2")
        base2.add_resource("login").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - safe path
        )

        # PRIVATE suppression still applies via this nested variable chain.
        private_api = apigw.RestApi(self, "PrivateApi2",
            endpoint_configuration=apigw.EndpointConfiguration(
                types=[apigw.EndpointType.PRIVATE]
            )
        )
        nested_base = private_api.root.add_resource("nested_base")
        nested_base.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - VPC-only API
        )


class AddMethodOnRootDirectly:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # add_method called directly on api.root — qualifier of add_method's callee
        # is a QualifiedExpression (api.root), so resolveToAddResourceCall returns null.
        # Path resolution yields empty; Gate 1 still fires for state-changing methods.
        api.root.add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Non state-changing method on api.root → no Gate 1, no path → Compliant.
        api.root.add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - no resource path resolved
        )


class HttpMethodMissing:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # Authorization is NONE but http_method argument is absent entirely.
        # Gate 1 cannot fire; path "users" is neutral → Compliant.
        api.root.add_resource("users").add_method(
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - http_method omitted
        )

        # No http_method but sensitive path → Gate 2 fires.
        api.root.add_resource("admin").add_method(
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


class ApiReassigned:
    def __init__(self):
        # api is reassigned, so singleAssignedValue returns null.
        # resolveToRestApiCall stays at the Name expression (not a CALL_EXPR)
        # and returns empty — PRIVATE suppression cannot be confirmed.
        api = apigw.RestApi(self, "ApiA")
        api = apigw.RestApi(self, "ApiB")
        api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


class ApiInstantiatedInline:
    def __init__(self):
        # apigw.RestApi(...) used inline, no variable.
        # qe.qualifier() in resolveRestApi is a CALL_EXPR (not a Name),
        # exercising the non-NAME branch of resolveToRestApiCall.
        apigw.RestApi(self, "InlineApi").root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


class EndpointConfigurationViaVariable:
    def __init__(self):
        # endpoint_configuration argument is a Name (config), not a direct
        # EndpointConfiguration(...) call. The predicate's CALL_EXPR check
        # fails on the Name expression in the flow.
        config_name = "regional"
        api = apigw.RestApi(self, "VarApi", endpoint_configuration=config_name)
        api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # endpoint_configuration is a CALL_EXPR but NOT EndpointConfiguration.
        api2 = apigw.RestApi(self, "WrongCallApi", endpoint_configuration=apigw.AuthorizationType.NONE)
        api2.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


class EndpointConfigurationWithoutTypes:
    def __init__(self):
        # EndpointConfiguration without "types" argument → cascade short-circuits.
        api = apigw.RestApi(self, "NoTypesApi",
            endpoint_configuration=apigw.EndpointConfiguration()
        )
        api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


class NonPathLiteralResource:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # path_part argument is a dynamic expression (not a string literal) — path stays empty.
        api.root.add_resource(get_dynamic_path()).add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
        api.root.add_resource(get_dynamic_path()).add_method(
            "GET",
            authorization_type=apigw.AuthorizationType.NONE  # Compliant - path unresolvable, safe method
        )


def get_dynamic_path():
    return external_value


class ResourceReassigned:
    def __init__(self):
        api = apigw.RestApi(self, "MyApi")

        # Variable holding a Resource is reassigned — singleAssignedValue returns null
        # for the qualifier of add_method, hitting the assigned==null branch of
        # resolveToAddResourceCall. Gate 1 still raises for state-changing methods.
        res = api.root.add_resource("first")
        res = api.root.add_resource("admin")
        res.add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Variable reassigned, then used as receiver of a nested add_resource —
        # exercises the assigned==null branch within resolveRestApi's NAME branch.
        nested_var = api.root.add_resource("a")
        nested_var = api.root.add_resource("b")
        nested_var.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


class EndpointConfigurationFromCall:
    def __init__(self):
        # endpoint_configuration is a CALL_EXPR but its FQN is not
        # apigw.EndpointConfiguration — the predicate returns false and suppression is skipped.
        api = apigw.RestApi(self, "FromCallApi", endpoint_configuration=build_some_config())
        api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )


def build_some_config():
    return None


class ApiAndResourceAliasedViaName:
    def __init__(self):
        # api is assigned to another Name (alias), not directly to a CALL_EXPR.
        # resolveToRestApiCall sees assigned != null but resolved.is(CALL_EXPR) is false.
        real_api = apigw.RestApi(self, "RealApi")
        api = real_api
        api.root.add_resource("admin").add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )

        # Resource alias: variable holding a Resource is reassigned via another Name.
        # In resolveToAddResourceCall, assigned != null but is not a CALL_EXPR.
        real_res = real_api.root.add_resource("inner")
        res_alias = real_res
        res_alias.add_method(
            "POST",
            authorization_type=apigw.AuthorizationType.NONE  # NonCompliant{{Ensure this API route requires authentication.}}
        )
