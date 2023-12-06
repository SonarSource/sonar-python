from graphql_server.flask import GraphQLView


app.add_url_rule("/api",
    view_func=GraphQLView.as_view(  # FN: SONARPY-1584
        name="api",
        schema=schema,
    )
)
