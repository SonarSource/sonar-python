from .cursor import Cursor
class Connection:
    def cursor(self) -> Cursor:
        ...
