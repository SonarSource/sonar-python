from .cursor import Cursor
class Connection:
    def __enter__(self) -> "Connection":
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ...

    def cursor(self) -> Cursor:
        ...
