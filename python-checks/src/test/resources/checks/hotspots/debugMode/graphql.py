from flask import Flask
from flask_graphql import GraphQLView

app.add_url_rule(
    '/graphql',
    view_func=GraphQLView.as_view(
        'graphql',
        schema=schema,
        graphiql=True # Noncompliant
    )
)

app.add_url_rule(
    '/graphql2',
    view_func=GraphQLView.as_view(
        'graphql2',
        schema=schema,
        graphiql=False
    )
)

# Per default GraphiQL is not enabled
app.add_url_rule(
    '/graphql3',
    view_func=GraphQLView.as_view(
        'graphql3',
        schema=schema
    )
)

class OverriddenView(GraphQLView):
    def new_func(self):
        return 1

app.add_url_rule(
    '/graphql4',
    view_func=OverriddenView.as_view(
        'graphql4',
        schema=schema,
        graphiql=True, # Noncompliant
    )
)

def assigned_value():
    view_func = GraphQLView.as_view(
        'graphql4',
        schema=schema,
        graphiql=True,  # Noncompliant
    )
    app.add_url_rule(
        '/graphql4',
        view_func=view_func
    )

    graphiql = True
    view_func = GraphQLView.as_view(
        'graphql4',
        schema=schema,
        graphiql=graphiql,  # Noncompliant
    )
    app.add_url_rule(
        '/graphql4',
        view_func=view_func
    )

def unused_view():
    GraphQLView.as_view(
        'graphql4',
        schema=schema,
        graphiql=True,  # Noncompliant
    )
