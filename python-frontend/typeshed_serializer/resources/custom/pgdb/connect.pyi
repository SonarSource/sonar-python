from .connection import Connection


def connect(dsn: str | None = None,
            user: str | None = None, password: str | None = None,
            host: str | None = None, database: str | None = None,
            **kwargs: Any) -> Connection:
    ...
