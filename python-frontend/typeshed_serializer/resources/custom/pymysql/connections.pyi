from .cursors import Cursor

class Connection:
    def cursor(self) -> Cursor:
        ...

def connect(dsn: str | None = None,
            user: str | None = None, password: str | None = None,
            host: str | None = None, database: str | None = None,
            **kwargs: Any) -> Connection:
    ...
