def flask_graphql_middleware_non_compliant(schema, some_middleware, another_middleware, get_middlewares):
    from flask_graphql import GraphQLView

    GraphQLView.as_view(  # Noncompliant {{Disable introspection on this "GraphQL" server endpoint.}}
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        schema=schema,
        name="introspection",
        graphiql=True,
        middleware=[some_middleware]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=(some_middleware, another_middleware)
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[get_middlewares()]
    )

    class OverriddenView(GraphQLView):
        pass

    OverriddenView.as_view(  # Noncompliant
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[]
    )

def overridden_graphql_server_noncompliant():
    from graphql_server.flask import GraphQLView
    class OverriddenView(GraphQLView):
        pass

    OverriddenView.as_view(  # Noncompliant
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[]
    )


def flask_graphql_middleware_compliant(schema, some_middleware, introspection_middleware, get_introspection_middlewares):
    def compliant_import():
        from a_module import GraphQLView
        GraphQLView.as_view(
            name="introspection",
            schema=schema,
            graphiql=True,
        )

    from flask_graphql import GraphQLView
    from a_module import IntrospectionMiddleware

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[some_middleware, introspection_middleware],
    )
    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=(some_middleware, introspection_middleware),
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[IntrospectionMiddleware],
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[get_introspection_middlewares()],
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=some_middleware, # if it is not a list or a tuple we should not raise an issue
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=some_middleware, # if it is not a list or a tuple we should not raise an issue
    )

    class IntrospectionCustomMiddleware:
            ...
    my_custom_middleware = IntrospectionCustomMiddleware()

    GraphQLView.as_view(
            name="introspection",
            schema=schema,
            graphiql=True,
            middleware=[my_custom_middleware]
        )

    def create_middleware():
        return IntrospectionCustomMiddleware()

    # FP this case is hard to detect and is more in the DBD territory
    GraphQLView.as_view( # Noncompliant
            name="introspection",
            schema=schema,
            graphiql=True,
            middleware=[create_middleware()]
        )

def flask_graphql_validation_rules_non_compliant(schema, some_rule):
    from flask_graphql import GraphQLView

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[],
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=(some_rule,)
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[some_rule]
    )

    class UnsafeCustomMiddleware:
            ...
    my_custom_middleware = UnsafeCustomMiddleware()

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
            name="introspection",
            schema=schema,
            graphiql=True,
            middleware=[my_custom_middleware]
        )

def flask_graphql_validation_rules_compliant(schema, some_rule, some_middleware, introspection_rule):
    from flask_graphql import GraphQLView
    import graphene
    from graphene.validation import DisableIntrospection
    from graphene.validation import DisableIntrospection as SafeRule
    from graphql.validation import NoSchemaIntrospectionCustomRule
    from graphql.validation import NoSchemaIntrospectionCustomRule as OtherSafeRule

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[NoSchemaIntrospectionCustomRule, some_rule]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[DisableIntrospection, some_rule]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[SafeRule, some_rule]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[some_middleware],
        validation_rules=[OtherSafeRule, some_rule]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[],
        validation_rules=[introspection_rule]
    )

def graphql_server_middleware_non_compliant(schema, some_middleware, CustomBackend, format_custom_error, some_module):
    from graphql_server.flask import GraphQLView

    GraphQLView.as_view(  # Noncompliant {{Disable introspection on this "GraphQL" server endpoint.}}
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[some_middleware]
    )

    igql_middlew = [
        some_middleware.IGQLProtectionMiddleware()
    ]

    GraphQLView.as_view( # Noncompliant
        'graphiql',
        schema=schema,
        backend=CustomBackend(),
        graphiql=True,
        middleware = igql_middlew,
        format_error=format_custom_error
    )

    GraphQLView.as_view( # Noncompliant
        'graphql',
        schema=schema,
        middleware=(some_module.unsafemiddleware,),
        backend=CustomBackend(),
        batch=True
    )

def graphql_server_middleware_compliant(schema, introspection_middleware, get_introspection_middlewares, CustomBackend, format_custom_error):
    from graphql_server.flask import GraphQLView

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[introspection_middleware],
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=get_introspection_middlewares(),
    )

    gql_middlew = [
        middleware.CostProtectionMiddleware(),
        middleware.DepthProtectionMiddleware(),
        middleware.IntrospectionMiddleware(),
        middleware.processMiddleware(),
    ]

    gql_middlew_qualified_expression = [
        middleware.IntrospectionMiddleware,
    ]

    GraphQLView.as_view( # OK
        'graphql',
        schema=schema,
        middleware=gql_middlew,
        backend=CustomBackend(),
        batch=True
    )

    GraphQLView.as_view( # OK
        'graphql',
        schema=schema,
        middleware=gql_middlew_qualified_expression,
        backend=CustomBackend(),
        batch=True
    )

    from a_module import some_function
    some_var = some_function()
    GraphQLView.as_view( # OK
        'graphiql',
        schema=schema,
        backend=CustomBackend(),
        graphiql=True,
        middleware=some_var,
        format_error=format_custom_error
    )

def graphql_server_validation_rules_non_compliant(schema, some_rule):
    from graphql_server.flask import GraphQLView
    import graphene

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[some_rule]
    )

def graphql_server_validation_rules_compliant(schema, some_rule, introspection_rule):
    from graphql_server.flask import GraphQLView
    import graphene
    from graphene.validation import DisableIntrospection
    from graphene.validation import DisableIntrospection as SafeRule
    from graphql.validation import NoSchemaIntrospectionCustomRule

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[NoSchemaIntrospectionCustomRule, some_rule]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[DisableIntrospection, some_rule]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[SafeRule, some_rule]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[introspection_rule]
    )
