
def flask_graphql_non_compliant(schema, some_middleware, introspection_middleware):
    from flask_graphql import GraphQLView
    from middleware import IntrospectionMiddleware, get_introspection_middleware

    GraphQLView.as_view(  # Noncompliant {{Disable introspection on this GraphQL server endpoint.}}
#   ^[el=+5;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[some_middleware, introspection_middleware]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=(some_middleware, introspection_middleware)
    )

    GraphQLView.as_view(  # Noncompliant
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[IntrospectionMiddleware]
    )


    GraphQLView.as_view(  # Noncompliant
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[get_introspection_middleware()]
    )




def flask_graphql_validation_rules_non_compliant(schema,compliant_middleware, some_rule, introspection_rule):
    from flask_graphql import GraphQLView
    import graphene
    from graphene.validation import DisableIntrospection
    from graphene.validation import DisableIntrospection as SensitiveRule
    from graphql.validation import NoSchemaIntrospectionCustomRule
    from graphql.validation import NoSchemaIntrospectionCustomRule as OtherSensitiveRule

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+7;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
        validation_rules=[]
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+7;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
        validation_rules=[NoSchemaIntrospectionCustomRule, some_rule]
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+7;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
        validation_rules=[DisableIntrospection, some_rule]
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+7;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
        validation_rules=[SensitiveRule, some_rule]
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+7;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
        validation_rules=[OtherSensitiveRule, some_rule]
    )


def flask_graphql_middleware_compliant(schema, compliant_rules, compliant_middleware, get_middlewares):
    def compliant_import():
        from a_module import GraphQLView
        GraphQLView.as_view(
            name="introspection",
            schema=schema,
            graphiql=True,
        )

    from flask_graphql import GraphQLView
    from graphql.validation import UniqueArgumentNamesRule
    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[
            compliant_middleware
        ],
        validation_rules = [ compliant_rules ]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=get_middlewares(),
        validation_rules = ( UniqueArgumentNamesRule , compliant_rules )
    )


def graphql_server_middleware_non_compliant(schema, some_middleware, introspection_middleware):
    from graphql_server.flask import GraphQLView

    GraphQLView.as_view(  # Noncompliant {{Disable introspection on this GraphQL server endpoint.}}
#   ^[el=+5;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[]
    )

    GraphQLView.as_view(  # Noncompliant
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[some_middleware, introspection_middleware]
    )

def graphql_server_validation_rules_non_compliant(schema,compliant_middleware, some_rule, introspection_rule):
    from graphql_server.flask import GraphQLView
    import graphene
    from graphene.validation import DisableIntrospection
    from graphene.validation import DisableIntrospection as SensitiveRule
    from graphql.validation import NoSchemaIntrospectionCustomRule
    from graphql.validation import NoSchemaIntrospectionCustomRule as OtherSensitiveRule

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+6;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+7;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
        validation_rules=[NoSchemaIntrospectionCustomRule, some_rule]
    )

    GraphQLView.as_view(  # Noncompliant 
#   ^[el=+7;ec=5]
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[compliant_middleware],
        validation_rules=[SensitiveRule, some_rule]
    )

def graphql_server_middleware_compliant(schema, compliant_middleware, compliant_rules, get_middlewares):
    from graphql_server.flask import GraphQLView
    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=[
            compliant_middleware
        ],
        validation_rules = [ compliant_rules ]
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        graphiql=True,
        middleware=get_middlewares(),
        validation_rules = [ compliant_rules ]
    )
