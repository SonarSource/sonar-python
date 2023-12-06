from graphql_server.flask import GraphQLView

app.add_url_rule("/api",
    view_func=GraphQLView.as_view(  # No issue if the project doesn't contain models with circular references
        name="api",
        schema=schema,
    )
)
