def flask_graphql_non_compliant(schema, some_middleware, another_middleware, get_middlewares):
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


def flask_graphql_validation_rules_compliant(schema, some_rule, introspection_rule):
    from flask_graphql import GraphQLView
    import graphene
    from graphene.validation import DisableIntrospection
    from graphene.validation import DisableIntrospection as SafeRule
    from graphql.validation import NoSchemaIntrospectionCustomRule
    from graphql.validation import NoSchemaIntrospectionCustomRule as OtherSensitiveRule

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
        validation_rules=[OtherSensitiveRule, some_rule]
    )

    GraphQLView.as_view( 
        name="introspection",
        schema=schema,
        graphiql=True,
        validation_rules=[introspection_rule]
    )


def graphql_server_middleware_non_compliant(schema, some_middleware):
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


def graphql_server_middleware_compliant(schema, introspection_middleware, get_introspection_middlewares):
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
