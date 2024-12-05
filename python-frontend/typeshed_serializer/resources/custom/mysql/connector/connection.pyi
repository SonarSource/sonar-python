from .cursor import MySQLCursor
class MySQLConnection:
    def cursor(self) -> MySQLCursor:
        ...
