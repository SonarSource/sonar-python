from flask_sqlalchemy.extension import SQLAlchemy as Database
from graphql_server.flask import GraphQLView

db = Database()
class Model(db.Model):
    children = db.relationship("Child", backref="parent")

GraphQLView.as_view(name="api", schema=schema)  # Noncompliant
GraphQLView.as_view(name="api", schema=schema, middleware=[DepthLimitMiddleware])
