from .cursor import MySQLCursor
class MySQLConnection:
    def __enter__(self) -> "MySQLConnection":
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    def cursor(self) -> MySQLCursor:
        ...
