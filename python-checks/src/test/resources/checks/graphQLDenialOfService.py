from graphql_server.flask import GraphQLView
from graphene.validation import depth_limit_validator
from flask_sqlalchemy import SQLAlchemy


from flask_sqlalchemy import SQLAlchemy

def schema_backref(app):
    db = SQLAlchemy(app)
    class Example1(db.Model):
        __tablename__ = 'example1'
        uuid = db.Column(db.Integer, primary_key=True)
        # `backref` will necessarily create a circular reference: it create an implicite relationship named 'parent' in ChildBackref
        child = db.relationship('Child', backref='parent') # This relationship creates circular references
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{This relationship creates circular references.}}

    class Example2(db.Model):
        __tablename__ = 'example2'
        uuid = db.Column(db.Integer, primary_key=True)
        # `back_populates` will necessarily create a circular reference: it will raise an exception unless there is an explixit relationship named 'parent' in ChildBackPopulates
        child = db.relationship('Child', back_populates='parent') # This relationship creates circular references
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{This relationship creates circular references.}}


def non_compliant():
    app.add_url_rule("/api",
        view_func=GraphQLView.as_view(  # Noncompliant {{Change this code to limit the depth of GraphQL queries.}}
#                 ^^^^^^^^^^^^^^^^^^^
            name="api",
            schema=schema,
        )
    )

    app.add_url_rule("/api",
        view_func=GraphQLView.as_view(  # Noncompliant
            name="api",
            schema=schema,
            validation_rules=[unknown_call()]
        )
    )

    app.add_url_rule("/api",
        view_func=GraphQLView.as_view(  # Noncompliant
            name="api",
            schema=schema,
            middleware=[UnknownMiddleware]
        )
    )

    app.add_url_rule("/api",
        view_func=GraphQLView.as_view()  # Noncompliant
    )

    app.add_url_rule("/api",
        view_func=GraphQLView.as_view(middleware=[])  # Noncompliant
    )

    app.add_url_rule("/api",
        view_func=GraphQLView.as_view(validation_rules=[])  # Noncompliant
    )

    class OverriddenView(GraphQLView):
            pass

    OverriddenView.as_view(name="api", schema=schema)  # Noncompliant


def compliant():
    from graphene.validation import DepthLimitValidator as my_validator

    app.add_url_rule("/api",
        view_func=GraphQLView.as_view(
            name="api",
            schema=schema,
            validation_rules=[
               depth_limit_validator(10) # Safe validation rule
            ]
        )
    )

    app.add_url_rule("/api",
        view_func=GraphQLView.as_view(
            name="api",
            schema=schema,
            validation_rules=[
               my_validator # Safe validation rule
            ]
        )
    )


    app.add_url_rule(
        '/compliant/graphql',
        view_func=GraphQLView.as_view( # Compliant
            'compliant',
            schema=schema,
            middleware=[DepthProtectionMiddleware]  # Safe middleware
        )
    )


    app.add_url_rule(
        '/compliant/graphql',
        view_func=GraphQLView.as_view( # Compliant
            'compliant',
            schema=schema,
            middleware=[DepthProtectionMiddleware],  # Safe middleware
            validation_rules=[DepthLimitValidator]   # Safe validation rules
        )
    )


    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        middleware=some_middleware, # if it is not a list or a tuple we should not raise an issue
    )

    GraphQLView.as_view(
        name="introspection",
        schema=schema,
        validation_rules=some_middleware, # if it is not a list or a tuple we should not raise an issue
    )
