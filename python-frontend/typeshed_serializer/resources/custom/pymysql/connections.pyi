from .cursors import Cursor

class Connection:
    def __enter__(self) -> "Connection":
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    def cursor(self) -> Cursor:
        ...

def connect(dsn: str | None = None,
            user: str | None = None, password: str | None = None,
            host: str | None = None, database: str | None = None,
            **kwargs: Any) -> Connection:
    ...
