from sqlalchemy import and_, or_, column, tuple_, create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.sql.elements import BooleanClauseList, ColumnElement


# and_/or_ may return subclasses of their broad ColumnElement annotation.
def conjunction() -> BooleanClauseList:
    return and_(column("x") == 1, column("y") == 2)


def disjunction() -> BooleanClauseList:
    return or_(and_(column("x") == n, column("y") == n) for n in [1, 2])


def broad_annotation() -> ColumnElement:
    return and_(column("x") == 1, column("y") == 2)


# Genuine mismatches: tuple IN produces BinaryExpression, and a raw DBAPI
# connection is not a SQLAlchemy Connection.
def tuple_filter() -> BooleanClauseList:
    return tuple_(column("x"), column("y")).in_([(1, 2)])  # Noncompliant


def raw_connection() -> Connection:
    return create_engine("sqlite://").raw_connection()  # Noncompliant
